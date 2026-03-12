/* ─── INITIALIZE MAPS ──────────────────────────────────────────────────── */
var mapLeft  = L.map('map-left',  { zoomControl: false, attributionControl: false }).setView([17.3850, 78.4867], 11);
var mapRight = L.map('map-right', { zoomControl: false, attributionControl: false }).setView([17.3850, 78.4867], 11);

const DARK_TILES = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
L.tileLayer(DARK_TILES).addTo(mapLeft);
L.tileLayer(DARK_TILES).addTo(mapRight);

/* ─── MAP SYNC ──────────────────────────────────────────────────────────── */
function syncMaps(source, target) {
    source.on('move', function () {
        if (!target.isSyncing) {
            source.isSyncing = true;
            target.setView(source.getCenter(), source.getZoom(), { animate: false });
            source.isSyncing = false;
        }
    });
}
syncMaps(mapLeft,  mapRight);
syncMaps(mapRight, mapLeft);

/* ─── LOCATION HUD ──────────────────────────────────────────────────────── */
var hudEl     = document.getElementById('location-hud');
var placeEl   = document.getElementById('place-name');
var hideTimer = null;
var debounce  = null;

function updateLocation(lat, lon) {
    hudEl.classList.add('visible');
    placeEl.innerText = 'SCANNING SECTOR...';

    fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=14`)
        .then(r => r.json())
        .then(d => {
            if (d && d.address) {
                var name = d.address.suburb
                        || d.address.neighbourhood
                        || d.address.residential
                        || d.address.village
                        || d.address.city_district
                        || d.address.city
                        || 'UNKNOWN';
                placeEl.innerText = 'SECTOR: ' + name.toUpperCase();
            } else {
                placeEl.innerText = 'SECTOR: UNKNOWN';
            }
            clearTimeout(hideTimer);
            hideTimer = setTimeout(() => hudEl.classList.remove('visible'), 3000);
        })
        .catch(() => { placeEl.innerText = 'DATA ERROR'; });
}

mapLeft.on('moveend', function () {
    var c = mapLeft.getCenter();
    clearTimeout(debounce);
    debounce = setTimeout(() => updateLocation(c.lat, c.lng), 400);
});

/* ─── ANALYSIS STATE ────────────────────────────────────────────────────── */
var geoLayerLeft  = null;
var geoLayerRight = null;

/* ─── STATUS HELPER ─────────────────────────────────────────────────────── */
function setStatus(msg, mode) {
    var bar  = document.getElementById('status-bar');
    var text = document.getElementById('status-text');
    bar.className = 'status-bar ' + mode;
    text.innerText = msg;
}

/* ─── ANIMATED COUNTER ──────────────────────────────────────────────────── */
function animateValue(elementId, target, decimals, duration) {
    var el   = document.querySelector('#' + elementId + ' .stat-value');
    if (!el) return;
    var start     = 0;
    var startTime = null;
    var suffix    = decimals > 0 ? '' : '';

    function step(timestamp) {
        if (!startTime) startTime = timestamp;
        var progress = Math.min((timestamp - startTime) / duration, 1);
        // Ease out
        var ease   = 1 - Math.pow(1 - progress, 3);
        var current = (ease * target).toFixed(decimals);
        el.innerText = current;
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

/* ─── RENDER STATS ──────────────────────────────────────────────────────── */
function renderStats(stats, startYear, endYear) {
    // Update labels in the badges
    document.getElementById('label-past').innerText   = 'PAST ' + startYear;
    document.getElementById('label-modern').innerText = 'MODERN ' + endYear;

    // Animate stat cards
    animateValue('stat-start',  stats.area_start,  1, 1200);
    animateValue('stat-end',    stats.area_end,    1, 1200);
    animateValue('stat-growth', stats.area_growth, 1, 1400);
    animateValue('stat-pct',    stats.pct_change,  1, 1600);
}

/* ─── MAIN ANALYSIS FUNCTION ────────────────────────────────────────────── */
function runAnalysis() {
    var city      = document.getElementById('city-select').value;
    var startYear = document.getElementById('start-year').value;
    var endYear   = document.getElementById('end-year').value;
    var btn       = document.getElementById('scan-btn');
    var btnText   = document.getElementById('btn-text');
    var btnLoader = document.getElementById('btn-loader');

    // Button → loading state
    btn.disabled = true;
    btnText.classList.add('hidden');
    btnLoader.classList.remove('hidden');
    setStatus('PROCESSING SATELLITE FEED...', 'loading');

    fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ city, start_year: startYear, end_year: endYear })
    })
    .then(r => r.json())
    .then(data => {
        // Restore button
        btn.disabled = false;
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');

        if (data.success) {
            setStatus('VISUALIZATION ACTIVE', 'success');

            // Remove old layers
            if (geoLayerLeft)  { mapLeft.removeLayer(geoLayerLeft); }
            if (geoLayerRight) { mapRight.removeLayer(geoLayerRight); }

            // Center both maps
            mapLeft.setView( [data.coords.lat, data.coords.lon], 12);
            mapRight.setView([data.coords.lat, data.coords.lon], 12);

            // Add new tile layers
            geoLayerLeft  = L.tileLayer(data.url_left).addTo(mapLeft);
            geoLayerRight = L.tileLayer(data.url_right).addTo(mapRight);

            // Trigger location HUD
            updateLocation(data.coords.lat, data.coords.lon);

            // Render stats panel
            renderStats(data.stats, data.start_year, data.end_year);

        } else {
            setStatus('ERROR: ' + data.error, 'error');
        }
    })
    .catch(err => {
        btn.disabled = false;
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
        setStatus('NETWORK FAILURE', 'error');
        console.error(err);
    });
}