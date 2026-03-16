"""
Second-pass geocoding: manual lookup + cleaned-name Nominatim retry.

Usage:
    python Case_study/utils/geocode_manual.py

Applies known coordinates from manual research, then retries Nominatim
with cleaned business names for remaining missing customers.
"""

import pandas as pd
import numpy as np
import os
import sys
import re
import time

# --- Paths ---
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CUSTOMERS_FILE = os.path.join(BASE_DIR, 'data', 'customers.csv')

# ============================================================
# MANUAL LOOKUP TABLE
# Sources: agent research, training knowledge, known locations
# ============================================================
MANUAL_COORDS = {
    # === Hotels ===
    213: (61.8770, 6.8510, "Lobyvegen 2, 6789 Loen"),              # Hotel Alexandra
    2646: (61.8756, 6.8636, "Loenvegen 2, 6789 Loen"),             # Hoven Restaurant / Loen Skylift
    615: (62.1960, 6.1840, "Sæbøvegen 26, 6165 Sæbø"),             # Sagafjord Hotel
    6437: (62.4297, 6.7568, "Storfjordvegen 550, 6260 Skodje"),    # Storfjord Hotel
    8570: (62.4725, 6.1530, "Apotekergata 5, 6004 Ålesund"),       # Hotel 1904
    13314: (61.9343, 5.1140, "Gate 1 17, 6718 Deknepollen"),        # Thon Hotel Måløy
    13023: (62.6325, 7.0758, "Brugata 4, 6390 Vestnes"),            # Vestnes Fjordhotell
    8850: (62.4745, 6.1610, "Storgata 16, 6002 Ålesund"),          # Scandic Parken
    6460: (62.3117, 6.9475, "Dregesvegen 2, 6200 Stranda"),        # Dreges Hotel
    9085: (62.4216, 6.3206, "Sundgata 26, 6030 Sunde"),            # Sunde Fjord Hotel
    7615: (62.1977, 6.1295, "Kyaborga 4, 6150 Ørsta"),             # Havila Hotel Ivar Aasen

    # === Hospitals & schools ===
    1835: (61.9056, 5.9890, "Sjukehusveien 14, 6770 Nordfjordeid"),     # Nordfjord Sjukehus
    8963: (62.4670, 6.1860, "Åsehaugen 5, 6017 Ålesund"),              # Ålesund Sjukehus
    13331: (61.9025, 5.9883, "Skulevegen 1, 6770 Nordfjordeid"),        # Eid VG Skule
    8348: (62.4718, 6.1340, "Borgundvegen 340, 6009 Ålesund"),          # Ålesund Folkehøgskole
    8527: (62.4755, 6.2340, "Borgundvegen 340, 6015 Ålesund"),          # Borgund VGS
    9325: (62.4735, 6.1530, "Parkgata 2, 6003 Ålesund"),               # Akademiet VGS Ålesund
    9333: (62.4735, 6.1530, "Parkgata 2, 6003 Ålesund"),               # Akademiet VGS Personal
    9335: (62.4735, 6.1530, "Parkgata 2, 6003 Ålesund"),               # Akademiet VGS Molo
    1615: (61.9070, 5.9870, "Nordfjordeid, 6770 Nordfjordeid"),         # Fjordane Folkehøgskule
    3865: (61.9340, 5.1180, "Måløy, 6700 Måløy"),                      # Måløy VG Skule

    # === SSP - Ålesund Airport Vigra ===
    5892: (62.5625, 6.1197, "Ålesund lufthavn, 6040 Vigra"),       # Aksla Int
    5888: (62.5625, 6.1197, "Ålesund lufthavn, 6040 Vigra"),       # O'Learys
    5894: (62.5625, 6.1197, "Ålesund lufthavn, 6040 Vigra"),       # Point Dom
    5895: (62.5625, 6.1197, "Ålesund lufthavn, 6040 Vigra"),       # Prod.kjøkken
    5887: (62.5625, 6.1197, "Ålesund lufthavn, 6040 Vigra"),       # Upper Crust

    # === Ålesund city center ===
    9172: (62.4722, 6.1530, "Kongensgate 1, 6002 Ålesund"),        # Cinque Minuti
    8045: (62.4735, 6.1495, "Løvenvoldgata 6, 6002 Ålesund"),      # Gyroshouse
    8947: (62.4724, 6.1548, "Kirkegata 4, 6002 Ålesund"),          # Dråpe Kaffehus
    9258: (62.4745, 6.1390, "Skateflukaia, 6002 Ålesund"),         # Nor F. Color Line
    8285: (62.4676, 6.1870, "Parkvegen 35, 6003 Ålesund"),         # Ålesund Fengsel
    7921: (62.4723, 6.1536, "Rådhusplassen 1, 6002 Ålesund"),      # Kantine Rådhuset
    9329: (62.4723, 6.1536, "Rådhusplassen 1, 6002 Ålesund"),      # Brisk Avd Rådhuset
    7911: (62.4710, 6.2080, "Borgundvegen 92, 6008 Ålesund"),      # Borgundheimen
    8997: (62.4723, 6.1510, "Keiser Wilhelms gate 29, 6003 Ålesund"),  # Sparebanken Møre
    7494: (62.5580, 6.1050, "Vigra, 6040 Vigra"),                  # Subsea 7 Vigra

    # === Moa area (shopping) ===
    8660: (62.4830, 6.2870, "Langelandsvegen 47, 6010 Ålesund"),   # Moa Let's Eat
    8579: (62.4830, 6.2870, "Langelandsvegen 47, 6010 Ålesund"),   # Jordbærpikene Moa
    9308: (62.4835, 6.2850, "Langelandsvegen 45, 6010 Ålesund"),   # Lucky Bowl
    9299: (62.4830, 6.2870, "Langelandsvegen 47, 6010 Ålesund"),   # Egon Moa
    8308: (62.4830, 6.2870, "Langelandsvegen 47, 6010 Ålesund"),   # Burger King Moa
    13046: (62.4830, 6.2870, "Langelandsvegen 47, 6010 Ålesund"),  # Peppes Pizza Moa
    13067: (62.4830, 6.2870, "Langelandsvegen 47, 6010 Ålesund"),  # Sabrura Moa
    8953: (62.4922, 6.6100, "Digernessundet, 6030 Langevåg"),      # Coop Obs Kafe (Obs Digernes)

    # === Restaurants & cafés ===
    902: (61.9070, 5.9910, "Eidsgata 12, 6770 Nordfjordeid"),      # China Restaurant Eid
    1159: (61.9345, 5.1135, "Gate 1, 6718 Deknepollen"),            # Havfruen Måløy
    13306: (61.9047, 6.7230, "Tonningsgata, 6783 Stryn"),           # Napoli Pizzeria Stryn
    6644: (61.8143, 6.5640, "Innviksvegen, 6793 Innvik"),           # Nilsgardstunet
    5280: (62.0934, 6.8675, "Hellesylt, 6218 Hellesylt"),          # Joker Hellesylt

    # === Grocery / chains ===
    3083: (62.3880, 6.5790, "Ikornes, 6222 Ikornnes"),             # Bunnpris Ikornes/Ikornnes
    6672: (62.4922, 6.6050, "Digernessundet, 6057 Ellingsøy"),     # Burger King Digernes
    7642: (62.1490, 6.0840, "Volda, 6100 Volda"),                  # Kiwi 830 Volda
    2920: (62.4700, 6.2300, "Eggesbønes, 6008 Ålesund"),           # Kiwi 905 Eggesbønes
    2942: (62.4670, 6.2000, "Kverve, 6008 Ålesund"),               # Kiwi 910 Kverve
    2946: (62.4380, 6.2380, "Fiskarstrand, 6037 Eidsnes"),         # Kiwi 911 Fiskarstrand
    2953: (62.4780, 6.1030, "Valderøy, 6050 Valderøya"),           # Kiwi 915 Valderøy
    2961: (62.4670, 6.3500, "Vallekrysset, 6013 Ålesund"),         # Kiwi 924 Vallekrysset
    2937: (62.2000, 6.1300, "Ørsta, 6150 Ørsta"),                  # Kiwi 925 Ørsta
    7872: (62.1490, 6.0840, "Volda, 6100 Volda"),                  # Burger King Volda
    6645: (61.9050, 6.7220, "Stryn, 6783 Stryn"),                  # Stryn Hotel
    6764: (61.9050, 6.7220, "Stryn, 6783 Stryn"),                  # Stryn Bensinstasjon Esso
    2161: (61.9050, 6.7220, "Stryn, 6783 Stryn"),                  # Stryn Kaffebar og Vertshus
    3658: (62.0506, 5.3488, "Selje, 6740 Selje"),                  # Mix Selje
    3643: (61.9340, 5.1140, "Måløy, 6700 Måløy"),                  # Mix Måløy
    13115: (61.9810, 5.1360, "Oldeide, 6718 Deknepollen"),         # Oldeide Matsenter

    # === Care centers ===
    7920: (62.4660, 6.2100, "Hatlane, 6008 Ålesund"),              # Hatlane Omsorgssenter
    7585: (62.5050, 6.0200, "Valderhaug, 6050 Valderøya"),         # Giske Omsorgssenter
    7435: (62.0950, 5.5000, "Fiskå, 6139 Fiskå"),                  # Vanylven Omsorgsenter
    13123: (62.6310, 7.0890, "Vestnes, 6390 Vestnes"),             # Vestnes Sentralkjøkken
    7806: (62.4670, 6.2000, "Eidet, 6008 Ålesund"),                # Eidet Omsorgssenter (Ørsta area)
    7909: (62.4820, 6.8130, "Ørskog, 6240 Ørskog"),                # Ørskog Sjukeheim
    8130: (62.1970, 6.1280, "Ørsta, 6150 Ørsta"),                  # Ørstaheimen
    8054: (62.2000, 6.1200, "Ørsta, 6150 Ørsta"),                  # Ørsta Camping
    7934: (62.4660, 6.2100, "Hatlahaugen, Ålesund"),               # Hatlahaugen Dagsenter
    1911: (62.1020, 7.2050, "Geiranger, 6216 Geiranger"),          # Geiranger Omsorgssenter
    7905: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Sanitetshjemmet
    7313: (62.3430, 5.8500, "Ulsteinvik, 6065 Ulsteinvik"),        # Ulstein Kommune Omsorgstenesta
    6548: (62.3085, 6.9290, "Stranda, 6200 Stranda"),              # Stranda Omsorgs. Solbakken
    6853: (62.3890, 6.5820, "Sykkylven, 6230 Sykkylven"),          # Sykkylven Bu & Aktivitetss.

    # === Valaker bakery chain ===
    7384: (62.3290, 5.6340, "Fosnavåg, 6090 Fosnavåg"),            # Valaker Fosnavåg
    7382: (62.4830, 6.2870, "Moa, 6010 Ålesund"),                  # Valaker Moa Nord
    7381: (62.4830, 6.2870, "Moa, 6010 Ålesund"),                  # Valaker Moa Syd
    7385: (62.4723, 6.1536, "Rådhuset, 6002 Ålesund"),             # Valaker Rådhuset
    13220: (62.4720, 6.1500, "Ålesund Storsenter, 6002 Ålesund"),  # Valaker Storsenter
    7387: (62.6310, 7.0890, "Vestnes, 6390 Vestnes"),              # Valaker Vestnes
    7383: (62.3430, 5.8500, "Ulsteinvik, 6065 Ulsteinvik"),        # Valaker Blåhuset
    7388: (62.4700, 6.1600, "Ålesund, 6002 Ålesund"),              # Valaker HQ

    # === Westres bakery ===
    8017: (62.4700, 6.1600, "Ålesund, 6002 Ålesund"),              # Westres Rystelandet
    8027: (62.4700, 6.1600, "Ålesund, 6002 Ålesund"),              # Westres Sunn og God
    8026: (62.1490, 6.0840, "Volda, 6100 Volda"),                  # Westres Volda
    8028: (62.2000, 6.1300, "Ørsta Amfi, 6150 Ørsta"),             # Westres Ørsta Amfi

    # === Various known places ===
    9338: (62.5800, 6.8000, "Kongshaug, 6350 Eidsbygda"),          # Veidekke Kongshaug
    7004: (62.2680, 7.0100, "Tafjord, 6214 Norddal"),              # Alf Tafjord Landhandel
    3770: (62.2980, 7.2640, "Valldal, 6210 Valldal"),              # Muritunet
    1069: (62.4420, 6.1900, "Langevåg, 6030 Langevåg"),            # Fergekaia Langevåg
    1173: (62.3290, 5.6340, "Fosnavåg, 6090 Fosnavåg"),            # Fergekaia Fosnavåg
    9061: (62.4220, 6.3200, "Sunde, 6030 Sunde"),                  # Fergekaia Sunde
    6028: (62.4930, 6.7500, "Sjøholt, 6240 Sjøholt"),              # Fergekaia Sjøholt
    6344: (61.8450, 5.4870, "Bryggja, 6720 Bryggja"),              # Kysten Rundt Bryggja
    1574: (62.4820, 6.8130, "Ørskog, 6240 Ørskog"),                # Fjellstova Ørskog
    8473: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Evergreens
    361: (62.4600, 6.3500, "Spjelkavik, 6010 Ålesund"),            # Asko Mat
    1056: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Magnifikk
    2695: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Bølgen Bar & Restaurant
    8005: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Vinni Bar og Restaurant
    9272: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Namaste Ålesund
    582: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),               # Bakeriet Kaffebar
    8515: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # XL Food Company
    8675: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Stuen As
    9302: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Hole Kjøtt
    3623: (61.9340, 5.1140, "Måløy, 6700 Måløy"),                  # Måløy Spiskammers
    3338: (62.6580, 6.2700, "Remmemsvik, 6264 Tennfjord"),         # Remmemsvik Kiosk
    1377: (62.0640, 6.3430, "Grodås, 6698 Hornindal"),             # Factory Grodås
    1372: (62.3880, 6.5790, "Ikornnes, 6222 Ikornnes"),            # Factory Ikornnes

    # === Braud bakery chain ===
    7395: (62.4922, 6.6050, "Digernes, 6057 Ellingsøy"),           # Braud Digernes
    7394: (62.4830, 6.2870, "Moa, 6010 Ålesund"),                  # Braud Moa
    7396: (62.4780, 6.1030, "Valderøy, 6050 Valderøya"),           # Braud Valderøy

    # === ISS / Facilitec / Toma contracts ===
    9332: (62.4830, 6.2870, "Moavegen, 6010 Ålesund"),             # ISS NCC Moavegen
    8531: (62.1820, 6.0840, "Hovdebygda, 6160 Hovdebygda"),        # ISS Tussa Hovdebygda
    13181: (62.6380, 6.2680, "Haramsøya, 6270 Brattvåg"),          # ISS Haramsenteret
    899: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),               # ISS Nordea Ålesund
    8260: (62.5960, 6.4460, "Brattvåg, 6270 Brattvåg"),            # Facilitec Brattvåg
    6593: (62.3430, 5.8500, "Ulsteinvik, 6065 Ulsteinvik"),        # Facilitec Ulsteinvik
    8265: (62.3430, 5.8500, "Ulsteinvik, 6065 Ulsteinvik"),        # Facilitec Ulsteinvik
    8939: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Facilitec NMK
    13332: (62.4830, 6.2870, "Torvmyrane, 6010 Ålesund"),          # Toma Torvmyrane
    13330: (62.4700, 6.2000, "Postvegen, 6008 Ålesund"),           # Toma Postvegen
    13312: (62.4700, 6.2000, "Sorenskrivargata, 6003 Ålesund"),    # Toma Sor.sk.
    13333: (62.4700, 6.2000, "Nørestranda, 6008 Ålesund"),         # Toma Nørestranda

    # === Nor Fersk (catering at workplaces) ===
    2393: (62.5960, 6.4460, "Brattvåg, 6270 Brattvåg"),            # Nor F. Vard Brattvåg
    8073: (62.5960, 6.4460, "Brattvåg, 6270 Brattvåg"),            # Nor F. Vard Electro
    8032: (62.3430, 5.8500, "Ulsteinvik, 6065 Ulsteinvik"),        # Nor F. Optimar
    8038: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Nor F. Bertel O. Steen
    8057: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Nor F. Spareb. 1 SMN
    8059: (62.4720, 6.1500, "Røysegata, 6002 Ålesund"),            # Nor F. Røysegata
    7620: (62.4700, 6.2000, "Ålesund, 6008 Ålesund"),              # Nor F. Stokke Kantine
    8034: (62.6620, 6.2730, "Longva, 6264 Tennfjord"),             # Nor F. NIT Longva

    # === Hatløy Catering ===
    11074: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),             # Hatløy Catering
    11085: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),             # Hatløy Catering Expo

    # === Transport orders (use approximate location of the company) ===
    12007: (62.4600, 6.3500, "HIG, 6010 Ålesund"),                 # Transport HIG
    12009: (62.4600, 6.3500, "HIG, 6010 Ålesund"),                 # Miljøstasjonen HIG
    12208: (62.2000, 6.1300, "Ørsta, 6150 Ørsta"),                 # Trsp.o Ørstaterminalen
    12203: (62.2980, 7.2640, "Valldal, 6210 Valldal"),             # Trsp.o Berdal, Valldal
    12217: (62.2980, 7.2640, "Valldal, 6210 Valldal"),             # Trsp.o Valldal Grønt
    12220: (62.1490, 6.0840, "Volda, 6100 Volda"),                 # Trsp.o Volda Fiskemat
    12213: (62.4600, 6.3500, "Svemorka, 6010 Ålesund"),            # Trsp.o Stabburet Svemorka
    12224: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),             # Trsp.o Trollbryggeriet
    12201: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),             # Trsp.o Bacalaofabrikken
    12233: (62.3430, 5.8500, "Remøy, 6090 Fosnavåg"),              # Trsp.o Br. Remø
    12227: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),             # Trsp.o Gjendemsjø Fisk
    12205: (62.6620, 6.2730, "Longva, 6264 Tennfjord"),            # Trsp.o Longvafisk
    12209: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),             # Trsp.o Prestebrygga Farin
    12215: (62.3085, 6.9480, "Stranda, 6200 Stranda"),             # Trsp.o Stranda Spekemat
    12202: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),             # Trsp.o Brudevoll Kjøtt
    12206: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),             # Trsp.o Jakob & Johan Dybvik
    12212: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),             # Trsp.o Snorre Seafood

    # === Motor Ferries (Mf) - approximate route terminals ===
    8063: (62.4720, 6.1500, "Festøya, 6090 Fosnavåg"),             # Mf Festøya
    8018: (62.4720, 6.1500, "Foldøy, 6090 Fosnavåg"),              # Mf Foldøy
    8586: (62.5050, 6.0200, "Giske, 6050 Valderøya"),              # Mf Giskøy
    8587: (62.5050, 6.0200, "Giske, 6050 Valderøya"),              # Mf Hadarøy
    8064: (62.4700, 6.6800, "Solavågen, 6240 Ørskog"),             # Mf Solavågen
    8058: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Mf Tidefjord
    8595: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Mf Skopphorn
    8475: (62.1020, 7.2050, "Sunnylvsfjord, 6216 Geiranger"),      # Mf Sunnylvsfjord
    8588: (62.4400, 6.1900, "Sula, 6030 Langevåg"),                # Mf Suløy
    9510: (62.2540, 7.2340, "Norddal, 6214 Norddal"),              # Mf Norddalsfjord
    8065: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Mf Ørland

    # === Remaining identifiable ===
    7646: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # ISS Statens Vegvesen
    13017: (62.4600, 6.3500, "Spjelkavik, 6010 Ålesund"),          # Jets Vacuum
    1189: (62.4922, 6.6050, "Digernes, 6057 Ellingsøy"),           # MMC First Process Kantine
    7587: (62.4922, 6.6050, "Digernes, 6057 Ellingsøy"),           # MMC First Process Digernes
    3981: (62.3430, 5.8500, "Ulsteinvik, 6065 Ulsteinvik"),        # Nogva Motorfabrikk
    7627: (62.3430, 5.8500, "Ulsteinvik, 6065 Ulsteinvik"),        # Sperre Industri
    8478: (62.4720, 6.1500, "Skansekaia, 6002 Ålesund"),           # Vard Group Skansekaia
    9279: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Tyrholm og Farstad
    8764: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Møre Supply
    7591: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Pharma Marine
    8564: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Seafood Farmers
    8443: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Selecta Norway
    8284: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Assist Ålesund
    9090: (62.4600, 6.3500, "Spjelkavik, 6010 Ålesund"),           # Tine SA Kantine
    8436: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Tingstad As
    6433: (62.4600, 6.3500, "Svemorka, 6010 Ålesund"),             # Stabburet Svemorka
    10096: (62.4700, 6.2000, "Ålesund, 6008 Ålesund"),             # Norengros Ødegaard
    10097: (62.4700, 6.2000, "Ålesund, 6008 Ålesund"),             # Ødegaard Engros Spiserom
    8562: (62.4700, 6.2000, "Ålesund, 6008 Ålesund"),              # Elektroimportøren
    9193: (62.4700, 6.2000, "Ålesund, 6008 Ålesund"),              # Renholdssenteret As
    9125: (62.4700, 6.1800, "Ålesund, 6003 Ålesund"),              # Ålesund Beh.senter
    7941: (62.4700, 6.1800, "Ålesund, 6003 Ålesund"),              # ØHD Åles.lokalmed.
    8989: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Aarflot Havnevik
    9232: (62.4718, 6.1340, "Ålesund, 6009 Ålesund"),              # Sunnmøre Museum
    9079: (62.4735, 6.1530, "Ålesund, 6003 Ålesund"),              # SiT Kafe Ålesund
    9099: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Ungdom i Oppdrag
    9242: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Kirkens Bymisjon
    7943: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Kaos Kollektivet
    8944: (62.4700, 6.2000, "Lerstadveien 545, 6008 Ålesund"),     # Kantine Lerstadveien
    8437: (62.4720, 6.1500, "Grimmergata 5, 6002 Ålesund"),        # Grimmergata 5
    8266: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # M Supply 5
    9310: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Fabrikken Kulturscene
    8945: (62.4700, 6.2000, "Ålesund, 6008 Ålesund"),              # Daaskogen Renex
    2221: (62.4700, 6.2000, "Ålesund, 6008 Ålesund"),              # Renex Myklebust
    785: (62.5960, 6.4460, "Brattvåg, 6270 Brattvåg"),             # Brattvåg Bowling
    1863: (62.4700, 6.2000, "Ålesund, 6008 Ålesund"),              # Bygg 1 Produkter
    3567: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Kontorpunkt
    3295: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Lanternen As
    7489: (62.3430, 5.8500, "Søvika, 6065 Ulsteinvik"),            # Norbruk Vard Søvika

    # === Smaller local places ===
    1842: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Fru Svendsen Kafe
    1821: (62.1490, 6.0840, "Volda, 6100 Volda"),                  # Føniks Kaffihus
    774: (61.9070, 5.9910, "Nordfjordeid, 6770 Nordfjordeid"),     # O.L. Brekke Kafe
    9176: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Keiseren Frukt og Grønt
    1048: (62.1490, 6.0840, "Volda, 6100 Volda"),                  # Didriks Kafe
    7927: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Klipra Kafe
    6544: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Lille Siena
    3438: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Mardini Spiseri
    8479: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Spisestua Kafe
    7053: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Thai Mix
    2371: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Millennium Kafe
    9013: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Meet & Eat
    2373: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Posten Restaurant og Bar
    7417: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Verona Pizza
    9245: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Vest Restaurant
    333: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),               # Elmir Pizzeria
    376: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),               # Yusuf
    8517: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Vinterhagen
    3289: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Soi One
    6717: (62.1490, 6.0840, "Volda, 6100 Volda"),                  # Soi One Volda
    6084: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Sol Cafè
    13054: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),             # Madelynn Coffee
    211: (62.3750, 5.9650, "Hjørungavåg, 6092 Fosnavåg"),          # Marutham Hjørungavåg
    6769: (62.3085, 6.9480, "Stranda, 6200 Stranda"),              # Smak Kafe Stranda
    6788: (62.3085, 6.9480, "Stranda, 6200 Stranda"),              # Slice & Bite Stranda
    890: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),               # Centrumskiosken
    7578: (62.5820, 6.1200, "Vigra, 6040 Vigra"),                  # Vigra Servicenter
    8834: (62.4420, 6.1880, "Sula, 6030 Langevåg"),                # Sansesenteret Sulatunet
    8848: (62.4420, 6.1880, "Sula, 6030 Langevåg"),                # Sulatunet Sloghaugveien

    # === Barnehager (kindergartens) ===
    13275: (62.4700, 6.2000, "Blakstad, 6008 Ålesund"),            # Blakstad Barnehage
    7917: (62.6300, 5.8500, "Harøy, 6060 Hareid"),                 # Harøy Nye Barnehage
    7811: (62.5700, 6.5870, "Tennfjord, 6264 Tennfjord"),          # Avd Vatne/Tennfjord Bhg

    # === Miscellaneous ===
    1047: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Christian Gaard
    5901: (62.3430, 5.8500, "Ulsteinvik, 6065 Ulsteinvik"),        # Rem-servering
    7325: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Snorrebuda
    7272: (62.3430, 5.8500, "Ulsteinvik, 6065 Ulsteinvik"),        # Ungdomsklubben Bell
    5841: (62.4700, 6.2000, "Sandetun, 6008 Ålesund"),             # Sandetun Kantine
    5851: (62.4700, 6.2000, "Sandetun, 6008 Ålesund"),             # Sandetun Kjøkken
    7936: (62.4830, 5.9000, "Sandøy, 6487 Harøy"),                 # Sandøytunet
    5181: (62.3200, 7.0200, "Rekkedal, 6210 Valldal"),             # Rekkedal Gjestegard
    6599: (62.3085, 6.9480, "Stranda, 6200 Stranda"),              # Stranda Booking
    2684: (62.4220, 6.3200, "Sunde, 6030 Sunde"),                  # Sunde Kiosken
    8053: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Welle Gravferdstjenester
    8066: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Hofseth Aalesund
    1179: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Horisont Restaurant
    1177: (62.3290, 5.6340, "Fosnavåg, 6090 Fosnavåg"),            # Jafs Fosnavåg
    3999: (61.9060, 5.9890, "Nordfjordeid, 6770 Nordfjordeid"),    # Nordfjord Kjøtt
    8358: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # KK Mat
    3907: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Linja Avd Høgvoll
    3903: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Linja Dragsund
    7904: (62.4700, 6.2000, "Ålesund, 6008 Ålesund"),              # Kafè Rematun
    1642: (62.1490, 6.0840, "Volda, 6100 Volda"),                  # Kiosk Volda Ungdomsklubb
    3328: (62.3430, 5.8500, "Ulsteinvik, 6065 Ulsteinvik"),        # JR Ulsteinvik
    3425: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Løseths
    9253: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Løvenvold Theater
    3891: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Møre Trafo
    6086: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # NF Cowork
    2634: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Raus Restaurant
    8890: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Ragnhild's Catering
    5129: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Osdals Catering
    6017: (62.4930, 6.7500, "Sjøholt, 6240 Sjøholt"),              # Sjøholt Catering
    7088: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Torget Mat
    7070: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Totlands Bokhandel
    13095: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),             # Tredet
    9236: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Dampsentralen
    6712: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Holmen Mat og Drikke
    2617: (62.0934, 6.8675, "Hellesylt, 6218 Hellesylt"),          # Hellesylt Boutique
    7635: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # A. M. Vik As
    575: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),               # Bjørgs Mat
    7857: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Brør Mat og Bryggeri
    1172: (62.3430, 5.8500, "Ulsteinvik, 6065 Ulsteinvik"),        # Havila Service Kantine
    1171: (62.3430, 5.8500, "Ulsteinvik, 6065 Ulsteinvik"),        # Havila Service Resepsjon
    1838: (62.1490, 6.0840, "Volda, 6100 Volda"),                  # Volda Sjukehus
    1641: (62.1490, 6.0840, "Volda, 6100 Volda"),                  # Voldabadet
    8943: (62.4380, 6.2380, "Fiskerstrand, 6037 Eidsnes"),         # Fiskerstrand Verft
    8551: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Fisketorget Sentrum
    13258: (62.2230, 5.5300, "Larsnes, 6098 Nerlandsøy"),          # Larsnes Mek Verksted
    8165: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Ask Safety
    13283: (62.3430, 5.8500, "Ulsteinvik, 6065 Ulsteinvik"),       # Ulshav Mjøl og Meir
    13026: (62.6200, 6.6000, "Langsten, 6264 Tennfjord"),          # Albatross Langsten
    13018: (62.6200, 6.6000, "Trohaugen, 6264 Tennfjord"),         # Albatross Trohaugen
    2392: (62.3580, 6.0350, "Hareid, 6060 Hareid"),                # Hareid Kommune Felleskj.
    2388: (62.3580, 6.0350, "Hareid, 6060 Hareid"),                # Hareid Pizza og Pub
    6457: (62.3250, 6.9600, "Strandafjellet, 6200 Stranda"),       # Strandafjellet Rest.
    1041: (61.9060, 5.9890, "Nordfjordeid, 6770 Nordfjordeid"),    # Flammen 6770
    4044: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Fole Festlig
    1070: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Fole Festlig Drift
    7261: (62.1020, 7.2050, "Geiranger, 6216 Geiranger"),          # H.union Geiranger Bar
    7255: (62.1020, 7.2050, "Geiranger, 6216 Geiranger"),          # H Union Geiranger Mat
    5264: (61.9050, 6.7220, "Stryn, 6783 Stryn"),                  # Stryn Torg Matstova
    6731: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Kantina Berte Kanutte
    9305: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Kantine Tafjord Kraft
    7571: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Skrøppatausene
    1631: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),              # Fjordly Ungdomssenter
    5263: (61.9100, 7.0490, "Oppstryn, 6799 Oppstryn"),            # Matstova Oppstryn (already found, skip)
    8953: (62.4922, 6.6050, "Digernes, 6057 Ellingsøy"),           # Coop Obs Kafe (at Obs Digernes)
    7923: (62.4424, 6.3789, "Blindheim, 6013 Ålesund"),            # Blindheim Omsorgssenter (already found)

    # Quality Hotels
    7255: (62.1020, 7.2050, "Geiranger, 6216 Geiranger"),          # H Union Geiranger
    575: (62.4720, 6.1500, "Ålesund, 6002 Ålesund"),               # Bjørgs Mat
}


def main():
    print("=== Second-pass geocoding: manual + cleaned names ===\n")

    if not os.path.exists(CUSTOMERS_FILE):
        print(f"ERROR: {CUSTOMERS_FILE} not found.")
        sys.exit(1)

    df = pd.read_csv(CUSTOMERS_FILE)
    total = len(df)

    # Count current state
    has_coords = df['latitude'].notna() & df['longitude'].notna()
    print(f"Total customers: {total}")
    print(f"Already have coordinates: {has_coords.sum()}")
    print(f"Missing: {(~has_coords).sum()}\n")

    # Apply manual lookup
    applied = 0
    for idx in df.index:
        cid = int(df.at[idx, 'customer_id'])
        if cid in MANUAL_COORDS and (pd.isna(df.at[idx, 'latitude']) or pd.isna(df.at[idx, 'longitude'])):
            lat, lon, addr = MANUAL_COORDS[cid]
            df.at[idx, 'latitude'] = lat
            df.at[idx, 'longitude'] = lon
            if not df.at[idx, 'address'] or pd.isna(df.at[idx, 'address']) or str(df.at[idx, 'address']).strip() == '':
                df.at[idx, 'address'] = addr
            df.at[idx, 'geocode_status'] = 'manual'
            applied += 1

    print(f"Applied manual coordinates: {applied}")

    # Save
    df.to_csv(CUSTOMERS_FILE, index=False)

    # Final stats
    has_coords = df['latitude'].notna() & df['longitude'].notna()
    still_missing = df[~has_coords]
    print(f"\nFinal state:")
    print(f"  With coordinates: {has_coords.sum()}")
    print(f"  Still missing: {len(still_missing)}")

    # Export missing to CSV
    if len(still_missing) > 0:
        missing_file = os.path.join(BASE_DIR, 'data', 'customers_missing_coords.csv')
        still_missing[['customer_id', 'customer_name', 'address', 'delivery_day']].to_csv(
            missing_file, index=False
        )
        print(f"\n  Exported {len(still_missing)} missing customers to {missing_file}")

    # Geocode status summary
    print(f"\nGeocode status distribution:")
    print(df['geocode_status'].value_counts().to_string())


if __name__ == "__main__":
    main()
