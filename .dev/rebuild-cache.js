#!/usr/bin/env node

/**
 * Cache rebuild script - supports both staging and production environments
 *
 * This script:
 * 1. Fetches all data from Supabase database
 * 2. Loads all geometry files from storage
 * 3. Processes service areas using timeline logic
 * 4. Uploads combined JSON blob to storage
 *
 * Environment Detection:
 * - STAGING=true env var → Uses staging tables/buckets
 * - Default → Uses production tables/buckets
 *
 * Run this whenever you update av_events or service_area_geometries data
 */

import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import { execSync } from 'child_process'
import { readFileSync, existsSync } from 'fs'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import area from '@turf/area'

// Load .env file for local development (GitHub Actions sets env vars directly)
if (!process.env.GITHUB_ACTIONS) {
  config()
}

// Require environment variables - no hardcoded secrets!
const supabaseUrl = process.env.SUPABASE_URL
const supabaseServiceKey = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_ANON_KEY

if (!supabaseUrl || !supabaseServiceKey) {
  console.error('❌ Missing required environment variables!')
  console.error('Please set SUPABASE_URL and SUPABASE_SERVICE_KEY')
  console.error('\nFor local development, create a .env file with:')
  console.error('SUPABASE_URL=your_supabase_url')
  console.error('SUPABASE_SERVICE_KEY=your_service_key')
  console.error('\nSee .env.example for template')
  process.exit(1)
}

// Detect environment: staging or production
const isStaging = process.env.STAGING === 'true'
const environment = isStaging ? 'staging' : 'production'

// Environment-specific configuration
const config_env = {
  eventsTable: isStaging ? 'av_events_staging' : 'av_events',
  geometriesTable: isStaging ? 'service_area_geometries_staging' : 'service_area_geometries',
  dataCacheBucket: isStaging ? 'staging-data-cache' : 'data-cache',
  geometriesBucket: isStaging ? 'staging-service-area-boundaries' : 'service-area-boundaries'
}

console.log('\n' + '='.repeat(70))
console.log(`🌍 ENVIRONMENT: ${environment.toUpperCase()}`)
console.log('='.repeat(70))
console.log(`📋 Configuration:`)
console.log(`   Events table: ${config_env.eventsTable}`)
console.log(`   Geometries table: ${config_env.geometriesTable}`)
console.log(`   Data cache bucket: ${config_env.dataCacheBucket}`)
console.log(`   Geometries bucket: ${config_env.geometriesBucket}`)
console.log('')

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function rebuildCache() {
  const startTime = Date.now()
  console.log('🚀 Starting cache rebuild...')

  try {
    // STEP 1: Fetch all database data
    console.log('📡 Fetching database data...')
    const [eventsResult, geometriesResult] = await Promise.all([
      supabase
        .from(config_env.eventsTable)
        .select('*')
        .eq('aggregate_type', 'service_area')
        .order('event_date', { ascending: true }),

      supabase
        .from(config_env.geometriesTable)
        .select('*')
        .order('created_at', { ascending: false })
    ])

    if (eventsResult.error) throw eventsResult.error
    if (geometriesResult.error) throw geometriesResult.error

    const events = eventsResult.data
    const geometriesMeta = geometriesResult.data

    console.log(`📊 Database data: ${events.length} events, ${geometriesMeta.length} geometries`)

    // STEP 2: Load all geometry files
    console.log('🌐 Loading geometry files...')
    const BATCH_SIZE = 8
    const geometriesWithData = []

    for (let i = 0; i < geometriesMeta.length; i += BATCH_SIZE) {
      const batch = geometriesMeta.slice(i, i + BATCH_SIZE)
      const batchNum = Math.floor(i/BATCH_SIZE) + 1
      const totalBatches = Math.ceil(geometriesMeta.length/BATCH_SIZE)

      console.log(`📦 Batch ${batchNum}/${totalBatches} (${batch.length} files)`)

      const batchResults = await Promise.all(
        batch.map(async (meta) => {
          try {
            const response = await fetch(meta.storage_url)
            if (!response.ok) {
              throw new Error(`HTTP ${response.status}`)
            }
            const geojsonData = await response.json()

            return {
              geometry_name: meta.geometry_name,
              display_name: meta.display_name,
              file_size: meta.file_size,
              created_at: meta.created_at,
              storage_url: meta.storage_url,
              geojson_data: geojsonData
            }
          } catch (error) {
            console.warn(`⚠️ Failed: ${meta.geometry_name}`)
            return {
              geometry_name: meta.geometry_name,
              display_name: meta.display_name,
              file_size: meta.file_size,
              created_at: meta.created_at,
              storage_url: meta.storage_url,
              geojson_data: null,
              error: error.message
            }
          }
        })
      )

      geometriesWithData.push(...batchResults)
    }

    const successful = geometriesWithData.filter(g => g.geojson_data)
    const failed = geometriesWithData.filter(g => !g.geojson_data)

    console.log(`✅ Geometries: ${successful.length} loaded, ${failed.length} failed`)

    // Create geometry lookup map with calculated areas
    const geometryMap = new Map()
    geometriesWithData.forEach(geo => {
      if (geo.geojson_data) {
        // Calculate area in square meters and convert to square miles
        const areaSquareMeters = area(geo.geojson_data)
        const areaSquareMiles = areaSquareMeters / 2589988.11 // 1 sq mi = 2,589,988.11 sq m
        geometryMap.set(geo.geometry_name, {
          geojson: geo.geojson_data,
          area_square_miles: Math.round(areaSquareMiles * 100) / 100 // Round to 2 decimals
        })
      }
    })

    // STEP 3: Process service areas
    console.log('⚙️ Processing service areas...')
    const serviceAreas = buildServiceAreasFromEvents(events, geometryMap)

    // STEP 3.5: Load news.csv if it exists
    console.log('📰 Loading news data...')
    let newsItems = []
    try {
      const __dirname = dirname(fileURLToPath(import.meta.url))
      const newsPath = resolve(__dirname, '..', 'news.csv')
      if (existsSync(newsPath)) {
        const newsCSV = readFileSync(newsPath, 'utf-8')
        const lines = newsCSV.trim().split('\n')
        if (lines.length > 1) {
          const header = lines[0].split(',')
          for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim()
            if (!line) continue
            // Simple CSV parse (handles quoted fields)
            const fields = []
            let current = '', inQuotes = false
            for (let j = 0; j < line.length; j++) {
              const ch = line[j]
              if (inQuotes) {
                if (ch === '"' && line[j + 1] === '"') { current += '"'; j++ }
                else if (ch === '"') { inQuotes = false }
                else { current += ch }
              } else {
                if (ch === '"') { inQuotes = true }
                else if (ch === ',') { fields.push(current); current = '' }
                else { current += ch }
              }
            }
            fields.push(current)
            const newsItem = {}
            header.forEach((col, idx) => { newsItem[col.trim()] = (fields[idx] || '').trim() })
            if (newsItem.headline) newsItems.push(newsItem)
          }
        }
      }
      console.log(`   Found ${newsItems.length} news items`)
    } catch (err) {
      console.warn('⚠️ Failed to load news.csv:', err.message)
    }

    // STEP 4: Create final data structure
    // Transform geometries to camelCase for frontend
    const geometriesForFrontend = geometriesWithData.map(geo => ({
      geometryName: geo.geometry_name,
      displayName: geo.display_name,
      fileSize: geo.file_size,
      createdAt: geo.created_at,
      storageUrl: geo.storage_url,
      geojsonData: geo.geojson_data,
      error: geo.error
    }));

    const cacheData = {
      metadata: {
        generated_at: new Date().toISOString(),
        version: Date.now(),
        export_stats: {
          total_events: events.length,
          total_geometries: geometriesMeta.length,
          loaded_geometries: successful.length,
          failed_geometries: failed.length,
          total_service_areas: serviceAreas.length,
          export_time_ms: Date.now() - startTime
        }
      },
      events: events,
      news: newsItems,
      geometries: geometriesForFrontend,
      service_areas: serviceAreas,
      date_range: {
        start: '2017-04-25T00:00:00+00:00',
        end: new Date().toISOString()
      }
    }

    // STEP 5: Upload to storage
    console.log('⬆️ Uploading to storage...')
    console.log(`   About to upload ${serviceAreas.length} service areas`)
    const jsonData = JSON.stringify(cacheData)
    const sizeMB = (jsonData.length / 1024 / 1024).toFixed(2)

    // Use timestamped filename to avoid caching issues (production only)
    if (!isStaging) {
      const timestamp = Date.now();
      const filename = `all-data-${timestamp}.json`;

      const { error: uploadError } = await supabase.storage
        .from(config_env.dataCacheBucket)
        .upload(filename, jsonData, {
          contentType: 'application/json',
          cacheControl: 'max-age=0, no-cache'
        })

      if (uploadError) throw uploadError
      console.log(`   Backup saved: ${filename}`)
    }

    // Upload as all-data.json
    await supabase.storage.from(config_env.dataCacheBucket).remove(['all-data.json'])
    const { error: mainUploadError } = await supabase.storage
      .from(config_env.dataCacheBucket)
      .upload('all-data.json', jsonData, {
        contentType: 'application/json',
        cacheControl: 'max-age=0, no-cache'
      })

    if (mainUploadError) throw mainUploadError

    const totalTime = Date.now() - startTime
    const publicUrl = `${supabaseUrl}/storage/v1/object/public/${config_env.dataCacheBucket}/all-data.json`

    console.log(`🎉 Cache rebuild complete!`)
    console.log(`   Environment: ${environment}`)
    console.log(`   Size: ${sizeMB}MB`)
    console.log(`   Time: ${totalTime}ms`)
    console.log(`   URL: ${publicUrl}`)

    if (failed.length > 0) {
      console.log(`⚠️ ${failed.length} geometries failed to load`)
    }

    return true

  } catch (error) {
    console.error('❌ Cache rebuild failed:', error)
    throw error
  }
}

// Helper function to transform snake_case to camelCase for frontend compatibility
function transformEventData(eventData) {
  const transformed = { ...eventData }

  // Transform snake_case fields to camelCase
  if (transformed.direct_booking !== undefined) {
    transformed.directBooking = transformed.direct_booking
    delete transformed.direct_booking
  }
  if (transformed.vehicle_types !== undefined) {
    transformed.vehicleTypes = transformed.vehicle_types
    delete transformed.vehicle_types
  }
  if (transformed.fleet_partner !== undefined) {
    transformed.fleetPartner = transformed.fleet_partner
    delete transformed.fleet_partner
  }
  if (transformed.geometry_name !== undefined) {
    transformed.geojsonPath = transformed.geometry_name
    delete transformed.geometry_name
  }
  if (transformed.service_model !== undefined) {
    transformed.serviceModel = transformed.service_model
    delete transformed.service_model
  }
  if (transformed.company_link !== undefined) {
    transformed.companyLink = transformed.company_link
    delete transformed.company_link
  }
  if (transformed.booking_platform_link !== undefined) {
    transformed.bookingPlatformLink = transformed.booking_platform_link
    delete transformed.booking_platform_link
  }
  if (transformed.expected_launch !== undefined) {
    transformed.expectedLaunch = transformed.expected_launch
    delete transformed.expected_launch
  }

  return transformed
}

// Helper function to parse inline coordinates from geojsonPath
function parseInlineCoordinates(geojsonPath) {
  if (!geojsonPath) return null
  // Check if it matches the inline coordinate format: "lng,lat" or "-lng,lat"
  const coordMatch = geojsonPath.match(/^(-?\d+\.?\d*),(-?\d+\.?\d*)$/)
  if (coordMatch) {
    return [parseFloat(coordMatch[1]), parseFloat(coordMatch[2])]
  }
  return null
}

// Service area processing logic
function buildServiceAreasFromEvents(events, geometryMap) {
  const currentServiceStates = new Map()
  const allStates = []

  const sortedEvents = [...events].sort((a, b) =>
    new Date(a.event_date).getTime() - new Date(b.event_date).getTime()
  )

  for (const event of sortedEvents) {
    const serviceId = event.aggregate_id
    const eventDate = new Date(event.event_date)
    const currentState = currentServiceStates.get(serviceId) || { isActive: false, status: null }

    if (event.event_type === 'service_testing') {
      const transformedData = transformEventData(event.event_data)
      const geojsonPath = transformedData.geojsonPath || event.event_data.geometry_name

      const geometryData = geometryMap.get(geojsonPath)
      const areaSquareMiles = geometryData?.area_square_miles || null

      // Parse inline coordinates if present
      const coordinates = parseInlineCoordinates(geojsonPath)

      const newState = {
        ...transformedData,
        id: `${serviceId}-${event.event_date}`,
        serviceId: serviceId,
        effectiveDate: eventDate.toISOString(),
        lastUpdated: eventDate.toISOString(),
        isActive: false,
        status: 'testing',
        geojsonPath: geojsonPath,
        area_square_miles: areaSquareMiles,
        ...(coordinates && { coordinates })
      }

      currentServiceStates.set(serviceId, newState)
      allStates.push(newState)

    } else if (event.event_type === 'service_announced') {
      const transformedData = transformEventData(event.event_data)
      const geojsonPath = transformedData.geojsonPath || event.event_data.geometry_name

      const geometryData = geometryMap.get(geojsonPath)
      const areaSquareMiles = geometryData?.area_square_miles || null

      // Parse inline coordinates if present
      const coordinates = parseInlineCoordinates(geojsonPath)

      // Merge previous state data with new event data (clone BEFORE closing to avoid including endDate)
      const prevState = JSON.parse(JSON.stringify(currentServiceStates.get(serviceId) || {}))

      // Always create a new state for service_announced to show timeline transitions
      // If there's an existing testing state, close it after cloning
      const lastState = allStates.filter(s => s.serviceId === serviceId && !s.endDate).pop()
      if (lastState) {
        lastState.endDate = eventDate.toISOString()
      }

      const newState = {
        ...prevState,
        ...transformedData,
        id: `${serviceId}-${event.event_date}`,
        serviceId: serviceId,
        effectiveDate: eventDate.toISOString(),
        lastUpdated: eventDate.toISOString(),
        isActive: false,
        status: 'announced',
        geojsonPath: geojsonPath,
        area_square_miles: areaSquareMiles,
        ...(coordinates && { coordinates })
      }

      currentServiceStates.set(serviceId, newState)
      allStates.push(newState)

    } else if (event.event_type === 'service_created') {
      const transformedData = transformEventData(event.event_data)
      const geojsonPath = transformedData.geojsonPath || event.event_data.geometry_name

      // Add calculated area from geometry
      const geometryData = geometryMap.get(geojsonPath)
      const areaSquareMiles = geometryData?.area_square_miles || null

      // Parse inline coordinates if present
      const coordinates = parseInlineCoordinates(geojsonPath)

      // Merge previous state data with new event data (clone BEFORE closing to avoid including endDate)
      const prevState = JSON.parse(JSON.stringify(currentServiceStates.get(serviceId) || {}))

      // Always create a new state for service_created to show timeline transitions
      // If there's an existing testing/announced state, close it after cloning
      const lastState = allStates.filter(s => s.serviceId === serviceId && !s.endDate).pop()
      if (lastState) {
        lastState.endDate = eventDate.toISOString()
      }

      const newState = {
        ...prevState,
        ...transformedData,
        id: `${serviceId}-${event.event_date}`,
        serviceId: serviceId,
        effectiveDate: eventDate.toISOString(),
        lastUpdated: eventDate.toISOString(),
        isActive: true,
        status: 'active',
        geojsonPath: geojsonPath,
        area_square_miles: areaSquareMiles,
        ...(coordinates && { coordinates })
      }

      currentServiceStates.set(serviceId, newState)
      allStates.push(newState)

    } else if (event.event_type === 'service_ended') {
      if (currentState.isActive) {
        const lastState = allStates.filter(s => s.serviceId === serviceId && !s.endDate).pop()
        if (lastState) {
          lastState.endDate = eventDate.toISOString()
        }
        currentServiceStates.set(serviceId, { ...currentState, isActive: false })
      }

    } else if (event.event_type === 'geometry_updated' || event.event_type === 'Service Area Change') {
      if (currentState.isActive || currentState.status === 'testing' || currentState.status === 'announced') {
        // Check if last state has same effectiveDate - if so, update in place instead of creating new state
        const lastState = allStates.filter(s => s.serviceId === serviceId && !s.endDate).pop()
        const lastStateDate = lastState ? new Date(lastState.effectiveDate).getTime() : 0
        const currentEventDate = eventDate.getTime()

        const newGeojsonPath = event.event_data.geometry_name || event.event_data.new_geometry_name || lastState?.geojsonPath
        const geometryData = geometryMap.get(newGeojsonPath)
        const areaSquareMiles = geometryData?.area_square_miles || null

        // Parse inline coordinates if present
        const coordinates = parseInlineCoordinates(newGeojsonPath)

        if (lastState && lastStateDate === currentEventDate) {
          // Same date - update existing state in place
          lastState.geojsonPath = newGeojsonPath
          lastState.area_square_miles = areaSquareMiles
          lastState.lastUpdated = eventDate.toISOString()
          if (coordinates) {
            lastState.coordinates = coordinates
          } else {
            delete lastState.coordinates
          }
          currentServiceStates.set(serviceId, lastState)
        } else {
          // Different date - create new state
          const newState = {
            ...JSON.parse(JSON.stringify(currentState)),
            id: `${serviceId}-${event.event_date}`,
            effectiveDate: eventDate.toISOString(),
            lastUpdated: eventDate.toISOString(),
            geojsonPath: newGeojsonPath,
            area_square_miles: areaSquareMiles,
            ...(coordinates && { coordinates })
          }

          if (lastState) {
            lastState.endDate = eventDate.toISOString()
          }

          currentServiceStates.set(serviceId, newState)
          allStates.push(newState)
        }
      }

    } else if (['service_updated', 'fares_policy_changed', 'access_policy_changed', 'vehicle_types_updated', 'platform_updated', 'supervision_updated', 'service_model_updated', 'fleet_partner_changed', 'direct_booking_updated'].includes(event.event_type)) {
      if (currentState.isActive || currentState.status === 'testing' || currentState.status === 'announced') {
        const shouldCreateNewState = ['fares_policy_changed', 'access_policy_changed', 'vehicle_types_updated', 'platform_updated', 'supervision_updated', 'service_model_updated', 'fleet_partner_changed', 'direct_booking_updated'].includes(event.event_type)

        if (shouldCreateNewState) {
          // Check if last state has same effectiveDate - if so, update in place instead of creating new state
          const lastState = allStates.filter(s => s.serviceId === serviceId && !s.endDate).pop()
          const lastStateDate = lastState ? new Date(lastState.effectiveDate).getTime() : 0
          const currentEventDate = eventDate.getTime()

          if (lastState && lastStateDate === currentEventDate) {
            // Same date - update existing state in place (using camelCase for frontend)
            if (event.event_type === 'fares_policy_changed') {
              lastState.fares = event.event_data.new_fares
            } else if (event.event_type === 'access_policy_changed') {
              lastState.access = event.event_data.new_access
            } else if (event.event_type === 'vehicle_types_updated') {
              lastState.vehicleTypes = event.event_data.new_vehicle_types || event.event_data.vehicle_types
            } else if (event.event_type === 'platform_updated') {
              lastState.platform = event.event_data.new_platform
            } else if (event.event_type === 'supervision_updated') {
              lastState.supervision = event.event_data.new_supervision
            } else if (event.event_type === 'service_model_updated') {
              lastState.serviceModel = event.event_data.new_service_model
            } else if (event.event_type === 'fleet_partner_changed') {
              lastState.fleetPartner = event.event_data.new_fleet_partner || event.event_data.fleet_partner
            } else if (event.event_type === 'direct_booking_updated') {
              lastState.directBooking = event.event_data.new_direct_booking
            }
            lastState.lastUpdated = eventDate.toISOString()
            currentServiceStates.set(serviceId, lastState)
          } else {
            // Different date - create new state
            const newState = {
              ...JSON.parse(JSON.stringify(currentState)),
              id: `${serviceId}-${event.event_date}`,
              effectiveDate: eventDate.toISOString(),
              lastUpdated: eventDate.toISOString()
            }

            // Apply field updates (using camelCase for frontend)
            if (event.event_type === 'fares_policy_changed') {
              newState.fares = event.event_data.new_fares
            } else if (event.event_type === 'access_policy_changed') {
              newState.access = event.event_data.new_access
            } else if (event.event_type === 'vehicle_types_updated') {
              newState.vehicleTypes = event.event_data.new_vehicle_types || event.event_data.vehicle_types
            } else if (event.event_type === 'platform_updated') {
              newState.platform = event.event_data.new_platform
            } else if (event.event_type === 'supervision_updated') {
              newState.supervision = event.event_data.new_supervision
            } else if (event.event_type === 'service_model_updated') {
              newState.serviceModel = event.event_data.new_service_model
            } else if (event.event_type === 'fleet_partner_changed') {
              newState.fleetPartner = event.event_data.new_fleet_partner || event.event_data.fleet_partner
            } else if (event.event_type === 'direct_booking_updated') {
              newState.directBooking = event.event_data.new_direct_booking
            }

            if (lastState) {
              lastState.endDate = eventDate.toISOString()
            }

            currentServiceStates.set(serviceId, newState)
            allStates.push(newState)
          }
        }
      }
    }
  }

  // Set default status for backward compatibility
  // Services without status field are assumed to be active (unless they have endDate)
  allStates.forEach(state => {
    if (!state.status) {
      state.status = state.endDate ? 'ended' : 'active'
    }
  })

  console.log(`   Created ${allStates.length} service area states from ${sortedEvents.length} events`)
  return allStates
}

// Run the rebuild
rebuildCache().catch(console.error)