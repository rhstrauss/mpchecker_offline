"""
Configuration and data paths for mpchecker.
"""
import os
from pathlib import Path

# Default data directory (~800 MB; override via $MPCHECKER_DATA)
_DEFAULT_DATA = Path.home() / 'mpchecker_data'
DATA_DIR = Path(os.environ.get('MPCHECKER_DATA', _DEFAULT_DATA))

# Subdirectories
ORBS_DIR = DATA_DIR / 'orbits'
SPICE_DIR = DATA_DIR / 'spice'
CACHE_DIR = DATA_DIR / 'cache'

# Orbit files
MPCORB_FILE = ORBS_DIR / 'MPCORB.DAT'
MPCORB_GZ   = ORBS_DIR / 'MPCORB.DAT.gz'
COMET_FILE  = ORBS_DIR / 'AllCometEls.txt'
OBSCODE_FILE = ORBS_DIR / 'ObsCodes.txt'

# pyoorb ephemeris file (installed with conda package)
_CONDA_PREFIX = os.environ.get('CONDA_PREFIX', '')
OORB_DATA = Path(os.environ.get('OORB_DATA',
    os.path.join(_CONDA_PREFIX, 'share', 'openorb') if _CONDA_PREFIX else ''))
OORB_EPHEM = OORB_DATA / 'de430.dat'

# SPICE kernels
SPICE_LSK   = SPICE_DIR / 'naif0012.tls'   # Leapseconds
SPICE_DE    = SPICE_DIR / 'de440s.bsp'      # Planetary positions (trimmed)
SPICE_PCK   = SPICE_DIR / 'pck00011.tpc'   # Planet constants
# Satellite kernels
SPICE_MAR   = SPICE_DIR / 'mar099.bsp'
SPICE_JUP   = SPICE_DIR / 'jup365.bsp'    # Galilean + inner moons (501-505, 514-516)
SPICE_JUP2  = SPICE_DIR / 'jup347.bsp'    # Irregular moons 506-572 + 55501-55526
SPICE_JUP3  = SPICE_DIR / 'jup348.bsp'    # Newest moons 55527-55530
SPICE_SAT   = SPICE_DIR / 'sat441.bsp'
SPICE_URA   = SPICE_DIR / 'ura116xl.bsp'
SPICE_NEP   = SPICE_DIR / 'nep097.bsp'
SPICE_PLU   = SPICE_DIR / 'plu060.bsp'

# Cached fast-lookup positions (numpy binary, updated by --update-cache)
CACHE_POSITIONS = CACHE_DIR / 'positions_{epoch_mjd_int}.npz'

# MPC data URLs
URL_MPCORB   = 'https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz'
URL_COMETS   = 'https://www.minorplanetcenter.net/iau/MPCORB/AllCometEls.txt'
URL_OBSCODES = 'https://www.minorplanetcenter.net/iau/lists/ObsCodes.html'

# NAIF SPICE URLs (NAIF anonymous FTP / HTTPS)
NAIF_BASE = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels'
SPICE_URLS = {
    'lsk':  f'{NAIF_BASE}/lsk/naif0012.tls',
    'de':   f'{NAIF_BASE}/spk/planets/de440s.bsp',
    'pck':  f'{NAIF_BASE}/pck/pck00011.tpc',
    'mar':  f'{NAIF_BASE}/spk/satellites/mar099.bsp',
    'jup':  f'{NAIF_BASE}/spk/satellites/jup365.bsp',
    'jup2': f'{NAIF_BASE}/spk/satellites/jup347.bsp',
    'jup3': f'{NAIF_BASE}/spk/satellites/jup348.bsp',
    'sat':  f'{NAIF_BASE}/spk/satellites/sat441.bsp',
    'ura':  f'{NAIF_BASE}/spk/satellites/ura116xl.bsp',
    'nep':  f'{NAIF_BASE}/spk/satellites/nep097.bsp',
    'plu':  f'{NAIF_BASE}/spk/satellites/plu060.bsp',
}

# Known NAIF IDs for planetary satellites to check, grouped by planet
# (NAIF body ID: display name)
SATELLITE_NAIF_IDS = {
    # Mars
    401: 'Phobos',
    402: 'Deimos',
    # Jupiter – inner moons (jup365.bsp: 501-505, 514-516)
    501: 'Io',
    502: 'Europa',
    503: 'Ganymede',
    504: 'Callisto',
    505: 'Amalthea',
    514: 'Thebe',
    515: 'Adrastea',
    516: 'Metis',
    # Jupiter – classic irregulars (jup347.bsp: 506-513)
    506: 'Himalia',
    507: 'Elara',
    508: 'Pasiphae',
    509: 'Sinope',
    510: 'Lysithea',
    511: 'Carme',
    512: 'Ananke',
    513: 'Leda',
    # Jupiter – outer irregulars (jup347.bsp: 517-572)
    517: 'Callirrhoe',
    518: 'Themisto',
    519: 'Magaclite',
    520: 'Taygete',
    521: 'Chaldene',
    522: 'Harpalyke',
    523: 'Kalyke',
    524: 'Iocaste',
    525: 'Erinome',
    526: 'Isonoe',
    527: 'Praxidike',
    528: 'Autonoe',
    529: 'Thyone',
    530: 'Hermippe',
    531: 'Aitne',
    532: 'Eurydome',
    533: 'Euanthe',
    534: 'Euporie',
    535: 'Orthosie',
    536: 'Sponde',
    537: 'Kale',
    538: 'Pasithee',
    539: 'Hegemone',
    540: 'Mneme',
    541: 'Aoede',
    542: 'Thelxinoe',
    543: 'Arche',
    544: 'Kallichore',
    545: 'Helike',
    546: 'Carpo',
    547: 'Eukelade',
    548: 'Cyllene',
    549: 'Kore',
    550: 'Herse',
    551: 'S/2010 J 1',
    552: 'S/2010 J 2',
    553: 'Dia',
    554: 'S/2016 J 1',
    555: 'S/2003 J 18',
    556: 'S/2011 J 2',
    557: 'Eirene',
    558: 'Philophrosyne',
    559: 'S/2017 J 1',
    560: 'Eupheme',
    561: 'S/2003 J 19',
    562: 'Valetudo',
    563: 'S/2017 J 2',
    564: 'S/2017 J 3',
    565: 'Pandia',
    566: 'S/2017 J 5',
    567: 'S/2017 J 6',
    568: 'S/2017 J 7',
    569: 'S/2017 J 8',
    570: 'S/2017 J 9',
    571: 'Ersa',
    572: 'S/2011 J 1',
    # Jupiter – newest discoveries (jup347.bsp: 55501-55526)
    55501: 'S/2003 J 2',
    55502: 'S/2003 J 4',
    55503: 'S/2003 J 9',
    55504: 'S/2003 J 10',
    55505: 'S/2003 J 12',
    55506: 'S/2003 J 16',
    55507: 'S/2003 J 23',
    55508: 'S/2003 J 24',
    55509: 'S/2011 J 3',
    55510: 'S/2018 J 2',
    55511: 'S/2018 J 3',
    55512: 'S/2021 J 1',
    55513: 'S/2021 J 2',
    55514: 'S/2021 J 3',
    55515: 'S/2021 J 4',
    55516: 'S/2021 J 5',
    55517: 'S/2021 J 6',
    55518: 'S/2016 J 3',
    55519: 'S/2016 J 4',
    55520: 'S/2018 J 4',
    55521: 'S/2022 J 1',
    55522: 'S/2022 J 2',
    55523: 'S/2022 J 3',
    55524: 'S/2025 J 1',
    55525: 'S/2017 J 10',
    55526: 'S/2017 J 11',
    # Jupiter – jup348.bsp: 55527-55530
    55527: 'S/2011 J 4',
    55528: 'S/2018 J 5',
    55529: 'S/2024 J 1',
    55530: 'S/2011 J 5',
    # Saturn (major satellites)
    601: 'Mimas',
    602: 'Enceladus',
    603: 'Tethys',
    604: 'Dione',
    605: 'Rhea',
    606: 'Titan',
    607: 'Hyperion',
    608: 'Iapetus',
    609: 'Phoebe',
    610: 'Janus',
    611: 'Epimetheus',
    612: 'Helene',
    613: 'Telesto',
    614: 'Calypso',
    615: 'Atlas',
    616: 'Prometheus',
    617: 'Pandora',
    618: 'Pan',
    # Uranus (major satellites)
    701: 'Ariel',
    702: 'Umbriel',
    703: 'Titania',
    704: 'Oberon',
    705: 'Miranda',
    706: 'Cordelia',
    707: 'Ophelia',
    708: 'Bianca',
    709: 'Cressida',
    710: 'Desdemona',
    711: 'Juliet',
    712: 'Portia',
    713: 'Rosalind',
    714: 'Belinda',
    715: 'Puck',
    # Neptune
    801: 'Triton',
    802: 'Nereid',
    803: 'Naiad',
    804: 'Thalassa',
    805: 'Despina',
    806: 'Galatea',
    807: 'Larissa',
    808: 'Proteus',
    # Pluto system
    901: 'Charon',
    902: 'Nix',
    903: 'Hydra',
    904: 'Kerberos',
    905: 'Styx',
}

# ---------------------------------------------------------------------------
# Dwarf planet satellite orbital elements
# (no SPICE kernels available; computed from published Keplerian elements)
#
# Elements are osculating Keplerian in the J2000 equatorial reference frame
# (ICRS), as published in the discovery/characterisation papers listed below.
# t_peri_mjd_tt : time of periapsis passage in MJD TT (approximate for objects
#                 without precise published value — positional error < a_sat,
#                 always << typical search radius of 30 arcmin).
# vmag_approx   : approximate apparent V magnitude (not H; already distance-
#                 corrected for the TNO's typical heliocentric distance).
# ---------------------------------------------------------------------------
DWARF_PLANET_SATELLITES = [
    # Dysnomia (136199 Eris I) — Brown et al. 2006, ApJ 639 L43
    dict(name='Dysnomia', primary_packed='D6199',
         a_au=2.491e-4, e=0.006,
         i_deg=78.29, Omega_deg=126.2, omega_deg=315.0,
         P_days=15.786, t_peri_mjd_tt=54610.4,
         vmag_approx=23.3),
    # Hi'iaka (136108 Haumea I) — Ragozzine & Brown 2009, AJ 137 4766
    dict(name="Hi'iaka", primary_packed='D6108',
         a_au=3.334e-4, e=0.050,
         i_deg=127.0, Omega_deg=206.7, omega_deg=69.0,
         P_days=49.462, t_peri_mjd_tt=54000.0,
         vmag_approx=20.7),
    # Namaka (136108 Haumea II) — Ragozzine & Brown 2009, AJ 137 4766
    dict(name='Namaka', primary_packed='D6108',
         a_au=1.715e-4, e=0.249,
         i_deg=113.0, Omega_deg=205.0, omega_deg=178.9,
         P_days=18.285, t_peri_mjd_tt=54000.0,
         vmag_approx=22.5),
    # Weywot (50000 Quaoar I) — Fraser et al. 2013, ApJ 774 L18
    dict(name='Weywot', primary_packed='50000',
         a_au=9.69e-5, e=0.140,
         i_deg=14.7, Omega_deg=44.0, omega_deg=197.0,
         P_days=12.438, t_peri_mjd_tt=55500.0,
         vmag_approx=21.8),
    # Vanth (90482 Orcus I) — Brown et al. 2010, AJ 139 847
    dict(name='Vanth', primary_packed='90482',
         a_au=6.02e-5, e=0.007,
         i_deg=105.0, Omega_deg=18.0, omega_deg=273.0,
         P_days=9.539, t_peri_mjd_tt=54610.0,
         vmag_approx=21.5),
    # Xiangliu (225088 Gonggong I) — Kiss et al. 2019, AJ 162 16
    dict(name='Xiangliu', primary_packed='M5088',
         a_au=1.61e-4, e=0.290,
         i_deg=99.0, Omega_deg=213.0, omega_deg=271.0,
         P_days=25.22, t_peri_mjd_tt=56200.0,
         vmag_approx=23.5),
    # Actaea (120347 Salacia I) — Grundy et al. 2011, Icarus 213 678
    dict(name='Actaea', primary_packed='C0347',
         a_au=3.76e-5, e=0.008,
         i_deg=23.5, Omega_deg=36.0, omega_deg=65.0,
         P_days=5.494, t_peri_mjd_tt=54610.0,
         vmag_approx=23.0),
    # MK2 (136472 Makemake I) — Parker et al. 2016, ApJ 825 L9
    dict(name='MK2', primary_packed='D6472',
         a_au=1.41e-4, e=0.000,
         i_deg=83.0, Omega_deg=100.0, omega_deg=0.0,
         P_days=12.40, t_peri_mjd_tt=57200.0,
         vmag_approx=26.0),
]

# Map satellite NAIF ID to its planet's SPK file key
# jup  = jup365.bsp (Galilean + inner: 501-505, 514-516)
# jup2 = jup347.bsp (irregulars: 506-513, 517-572, 55501-55526)
# jup3 = jup348.bsp (newest: 55527-55530)
_JUP_INNER = {501, 502, 503, 504, 505, 514, 515, 516}
_JUP2_IDS  = (set(range(506, 513+1)) | set(range(517, 572+1)) |
               set(range(55501, 55526+1)))
_JUP3_IDS  = {55527, 55528, 55529, 55530}

SATELLITE_KERNEL_MAP = {
    401: 'mar', 402: 'mar',
    **{n: 'jup'  for n in _JUP_INNER},
    **{n: 'jup2' for n in _JUP2_IDS},
    **{n: 'jup3' for n in _JUP3_IDS},
    **{600+i: 'sat' for i in range(1, 40)},
    **{700+i: 'ura' for i in range(1, 30)},
    **{800+i: 'nep' for i in range(1, 20)},
    **{900+i: 'plu' for i in range(1, 10)},
}
