# UniFi AI Surveillance Prototype

Proiect realizat pentru practica de domeniu - Universitatea din Craiova.

Student : Movileanu Robert-Marian

Prof. Indrumator : Hurezeanu Bogdan

Aplicatia reprezinta un prototip software pentru simularea capabilitatilor de Edge AI dintr-un sistem de supraveghere bazat pe arhitectura Ubiquiti UniFi Protect.

## Arhitectura sistemului
- Camere IP conectate prin PoE la un switch de retea, izolate pe VLAN dedicat (VLAN 10).
- Prelucrare locala pe baza modelului YOLOv8n pentru detectie in timp real de vehicule si persoane.
- Panou de control dezvoltat in Streamlit pentru vizualizarea fluxului video si a alertelor.

## Rulare proiect

1. Creare si activare mediu virtual:
```bash
python -m venv venv
venv\Scripts\activate

2. Instalare dependinte:
Bash
pip install -r requirements.txt

3. Lansare aplicatie
Bash
streamlit run app.py
