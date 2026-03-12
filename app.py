from flask import Flask, render_template, jsonify, request
import ee
import traceback

app = Flask(__name__)

# Initialize GEE
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

CITIES = {
    'Hyderabad': {'lat': 17.3850, 'lon': 78.4867},
    'Bangalore': {'lat': 12.9716, 'lon': 77.5946},
    'Mumbai':    {'lat': 19.0760, 'lon': 72.8777},
    'Delhi':     {'lat': 28.7041, 'lon': 77.1025},
    'Chennai':   {'lat': 13.0827, 'lon': 80.2707},
    'Kolkata':   {'lat': 22.5726, 'lon': 88.3639},
    'Pune':      {'lat': 18.5204, 'lon': 73.8567}
}

def get_satellite_image(year):
    """Fetch a clean, cloud-free annual median composite."""
    print(f"DEBUG: Processing Year {year}...")

    if year >= 2013:
        coll_id  = "LANDSAT/LC08/C02/T1_L2"
        bands_in = ['SR_B4', 'SR_B3', 'SR_B2', 'SR_B5', 'SR_B6']
    else:
        coll_id  = "LANDSAT/LE07/C02/T1_L2" if year == 2012 else "LANDSAT/LT05/C02/T1_L2"
        bands_in = ['SR_B3', 'SR_B2', 'SR_B1', 'SR_B4', 'SR_B5']

    bands_out  = ['red', 'green', 'blue', 'nir', 'swir']
    collection = ee.ImageCollection(coll_id).filterDate(f'{year}-01-01', f'{year}-12-31')

    if collection.size().getInfo() == 0:
        print(f"DEBUG: Year {year} empty – expanding window.")
        collection = ee.ImageCollection(coll_id).filterDate(f'{year-1}-01-01', f'{year+1}-12-31')

    def apply_scale_factors(image):
        optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        return image.addBands(optical, None, True)

    img = (collection
           .map(apply_scale_factors)
           .median()
           .select(bands_in, bands_out))

    return img


def get_urban_mask(img):
    """Classify urban pixels using spectral indices + JRC permanent water exclusion + morphological filtering."""
    ndbi  = img.normalizedDifference(['swir', 'nir'])   # Built-up Index
    ndvi  = img.normalizedDifference(['nir',  'red'])   # Vegetation Index

    # ── Spectral classification ───────────────────────────────────────────────
    is_urban_spectral = (ndbi.gt(-0.05)
                         .And(ndvi.lt(0.2)))

    # ── JRC Global Surface Water permanent water mask ─────────────────────────
    # 'occurrence' band = % of time a pixel was water from 1984–present (0–100).
    # Any pixel observed as water ≥ 10% of the time is treated as permanent water
    # and is unconditionally excluded from urban classification.
    # This correctly removes the Bay of Bengal, Buckingham Canal, rivers, and lakes
    # regardless of how they look on the specific year's satellite image.
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    permanent_water = jrc.select('occurrence').unmask(0).gt(10)  # 10% threshold
    not_water = permanent_water.Not()

    is_urban = is_urban_spectral.And(not_water)

    # ── Morphological Closing: remove noise + fill small gaps ─────────────────
    cleaned = (is_urban.focalMax(1.5, 'circle', 'pixels')
                       .focalMin(1.5, 'circle', 'pixels'))
    return cleaned.selfMask()


def compute_urban_area_km2(mask, region):
    """Calculate the total urban area in km² for a given binary mask + region."""
    area_image = mask.unmask(0).multiply(ee.Image.pixelArea())
    stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=100,          # ~100m pixel sampling for reasonable GEE speed
        maxPixels=1e9
    )
    area_m2 = stats.get('nd').getInfo()
    if area_m2 is None:
        return 0.0
    return round(area_m2 / 1_000_000, 2)  # Convert m² → km²


def get_comparison_layers(start_year, end_year, city_coords):
    # ─── 1. FETCH IMAGERY ────────────────────────────────────────────────────
    img_start = get_satellite_image(start_year)
    img_end   = get_satellite_image(end_year)

    # ─── 2. CLASSIFY URBAN ───────────────────────────────────────────────────
    urban_start = get_urban_mask(img_start)
    urban_end   = get_urban_mask(img_end)

    # ─── 3. GROWTH DETECTION ─────────────────────────────────────────────────
    # Pixels that became urban during the study period
    growth = urban_end.unmask(0).subtract(urban_start.unmask(0)).gt(0).selfMask()

    # ─── 4. STATISTICS (area in km²) ─────────────────────────────────────────
    point  = ee.Geometry.Point([city_coords['lon'], city_coords['lat']])
    region = point.buffer(30_000)  # 30 km radius study area

    area_start  = compute_urban_area_km2(urban_start, region)
    area_end    = compute_urban_area_km2(urban_end,   region)
    area_growth = round(area_end - area_start, 2)
    pct_change  = round(((area_end - area_start) / max(area_start, 1)) * 100, 1)

    # ─── 5. VISUALIZE ────────────────────────────────────────────────────────
    vis_params  = {'min': 0.0, 'max': 0.3, 'bands': ['red', 'green', 'blue'], 'gamma': 1.2}
    urban_vis   = {'palette': ['ff0000'], 'opacity': 0.75}  # Red  – existing
    growth_vis  = {'palette': ['ffff00'], 'opacity': 1.0}   # Yellow – new growth

    # LEFT: past satellite + past urban mask
    layer_left = img_start.visualize(**vis_params).blend(urban_start.visualize(**urban_vis))
    url_left   = layer_left.getMapId()['tile_fetcher'].url_format

    # RIGHT: present satellite + stable urban (red) + new growth (yellow)
    layer_right = (img_end.visualize(**vis_params)
                   .blend(urban_start.visualize(**urban_vis))
                   .blend(growth.visualize(**growth_vis)))
    url_right   = layer_right.getMapId()['tile_fetcher'].url_format

    return url_left, url_right, {
        'area_start':  area_start,
        'area_end':    area_end,
        'area_growth': area_growth,
        'pct_change':  pct_change
    }


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', cities=CITIES)


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data       = request.json
        start      = int(data['start_year'])
        end        = int(data['end_year'])
        city_name  = data['city']
        coords     = CITIES[city_name]

        url_left, url_right, stats = get_comparison_layers(start, end, coords)

        return jsonify({
            'success':    True,
            'url_left':   url_left,
            'url_right':  url_right,
            'coords':     coords,
            'stats':      stats,
            'start_year': start,
            'end_year':   end,
            'city':       city_name
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)