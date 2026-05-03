import urllib.request
import json
import time

locations = [
    ('Ambalantota', 'Hambantota', 'Southern'),
    ('Ehetuwewa', 'Kurunegala', 'North Western'),
    ('Giribawa', 'Kurunegala', 'North Western'),
    ('Gomarankadawala', 'Trincomalee', 'Eastern'),
    ('Karuwalagaswewa', 'Puttalam', 'North Western'),
    ('Lankapura', 'Polonnaruwa', 'North Central'),
    ('Mahawilachchiya', 'Anuradhapura', 'North Central'),
    ('Mihinthale', 'Anuradhapura', 'North Central'),
    ('Morawewa', 'Trincomalee', 'Eastern'),
    ('Musali', 'Mannar', 'Northern'),
    ('Palagala', 'Anuradhapura', 'North Central'),
    ('Sammanthurai', 'Ampara', 'Eastern'),
    ('Seruvila', 'Trincomalee', 'Eastern'),
    ('Thanamalwila', 'Monaragala', 'Uva'),
    ('Vavuniya South', 'Vavuniya', 'Northern'),
    ('Verugal', 'Trincomalee', 'Eastern'),
    ('Welioya', 'Mullaitivu', 'Northern')
]

for name, dist, prov in locations:
    q = f"{name} {dist} Sri Lanka".replace(" ", "%20")
    url = f"https://nominatim.openstreetmap.org/search?q={q}&format=json&limit=1"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req).read()
        data = json.loads(resp)
        if data:
            print(f'    {{"id": "{name}", "district": "{dist}", "province": "{prov}", "lat": {float(data[0]["lat"]):.4f}, "lon": {float(data[0]["lon"]):.4f}}},')
        else:
            print(f'# Failed for {name}')
    except Exception as e:
        print(f'# Error {name}: {e}')
    time.sleep(1)
