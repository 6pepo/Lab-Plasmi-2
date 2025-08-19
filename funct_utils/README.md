# funct_utils

Libreria Python per:
- Calcolo della larghezza a metà altezza (FWHM)
- Rilevamento dei picchi locali
- Rilevamento dei minimi locali (con filtro Savitzky-Golay)

## Installazione

```bash
pip install -e . --no-use-pep517 --no-build-isolation
```

## Uso

```python
from funct_utils import find_FWHM, find_peaks, find_min
```
