# Mbajk availability tracker

## Project Structure

- **data:**
  - *processed:* Procesirani podatki, pripravljeni za učenje.
  - *raw:* Prenešeni podatki v originalni obliki.

- **models:**
  - Naučeni in serializirani modeli, napovedi modelov ali povzetki modelov.

- **notebooks:**
  - Jupyter zvezki.

- **reports:**
  - Generirane datoteke analiz.
    - *figures:* Generirani grafi in slike, uporabljene pri analizi.

- **pyproject.toml:**
  - Datoteka, ki definira odvisnosti, verzije knjižnic...

- **src:**
  - *__init__.py:* Ustvari direktorij "src" kot Python modul.
  - **data:**
    - Skripte za prenos, procesiranje, itd. podatkov.
  - **models:**
    - Skripte za učenje napovednih modelov in uporabo modelov za napovedovanje.
  - **serve:**
    - Skripte za serviranje modelov v obliki spletnih storitev.
  - **visualization:**
    - Skripte za vizualizacijo.
