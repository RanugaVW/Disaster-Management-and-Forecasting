import sys
import re
sys.path.insert(0, '.')
from ds_locations import DS_LOCATIONS

existing = {d['id']: d for d in DS_LOCATIONS}

text = """Dimbulagala, Kandavalai, Maritimepattu, Welikanda, Medirigiriya, Horowpathana, Poonakary, Kantale, Kuchchaweli, Dehiattakandiya, Koralai Pattu South, Vavuniya, Koralai Pattu North, Manthai West, Karachchi, Mahiyanganaya, Kahatagasdigiliya, Medawachchiya, Kekirawa, Rambewa, Eravur Pattu
Vadamaradchchi East, Galenbidunuwewa, Thamankaduwa, Nanaddan, Oddusuddan, Nuwaragam Palatha Central, Muthur, Kebithigollewa, Thenmaradchi, Nochchiyagama, Higurakgoda, Galgamuwa, Mundel, Mannar Town, Padaviya, Vengalacheddikulam, Vavuniya North, Pachchilaipalli, Kinniya, Thirappane, Thalawa
Dompe, Polpitigama, Mahawilachchiya, Lankapura, Dambulla, Gomarankadawala, Sammanthurai, Mihinthale, Welioya, Verugal, Palagala, Musali, Karuwalagaswewa, Giribawa, Seruvila, Thanamalwila, Morawewa, Ambalantota, Vavuniya South, Ehetuwewa, Ududumbara
Ududumbara, Laggala, Lunugala, Kothmale East, Passara, Meegahakiula, Soranathota, Rideegama, Hanguranketa, Kandeketiya, Thalawakele, Medadumbara, Minipe, Deltota, Doluwa, Mathurata, Pathahewaheta, Ambanganga, Aranayake, Rattota, Walapane
Yatiyantota, Rideemaliyadda, Ibbagamuwa, Nuwara Eliya, Mawathagama, Ganga Ihala Korale, Ukuwela, Nildandahinna, Mawanella, Warakapola, Poojapitiya, Naula, Wilgamuwa, Hatharaliyadda, Yatawatta, Pathadumbara, Panvila, Polgahawela, Rambukkana, Udapalatha
Polpitigama, Mallawapitiya, Welimada, Ratnapura, Bulathkohipitiya, Yatinuwara, Matale, Kothmale West, Alawwa, Badulla, Bibile, Kuruvita, Kolonna, Udunuwara, Dompe, Akurana, Harispattuwa, Pasbagekorale, Dambulla, Pallepola, Norwood"""

requested = set(x.strip() for x in re.split(r'[,]+|\n', text))
requested = {x for x in requested if x and not x.startswith('Image') and 'Total' not in x}
print(f'Total unique requested: {len(requested)}')
missing = requested - set(existing.keys())
print(f'Missing from registry ({len(missing)}):')
for m in sorted(missing):
    print(f"  {m}")
