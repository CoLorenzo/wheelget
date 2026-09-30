# wheelget

Trova e scarica la wheel giusta per la tua GPU NVIDIA: legge la versione di CUDA
del driver installato e sceglie la release piu' recente del pacchetto compatibile
con quella CUDA.

- `wheelget url <pkg>` stampa il link diretto della wheel
- `wheelget get <pkg>` scarica la wheel nella directory corrente
- `wheelget get vllm` scarica anche la wheel di `torch` pinnata nel METADATA di
  vllm (es. `Requires-Dist: torch==2.13.0`), preferendo la stessa variante CUDA
  di vllm

Provider supportati: `vllm` (indice `wheels.vllm.ai`) e `torch`
(indice `download.pytorch.org/whl`, alias `pytorch`).

## Installazione

```bash
uv tool install .
```

(oppure `uv tool install git+<url-del-repo>` quando il repo e' pubblicato)

## Uso

```bash
# la CUDA viene rilevata da nvidia-smi / nvcc / CUDA_HOME
wheelget url vllm
wheelget get vllm

# forzare una CUDA (utile su macchine senza driver NVIDIA)
wheelget url torch --cuda 12.6
wheelget url torch --cuda cu130

# scaricare in una cartella specifica
wheelget get vllm -o ~/wheels

# fissare versione / variante / Python target
wheelget url torch 2.9.1 --variant cu128
wheelget url vllm 0.30.0 --python 3.12

# dato un .whl di torch, stampa la versione Python richiesta (solo quella)
wheelget torch-compability ~/wheels/torch-2.13.0+cu129-cp313-cp313-manylinux_2_28_x86_64.whl
# -> 3.13
```

Se non passi il percorso, `torch-compability` prende l'unico `torch*.whl`
presente nella directory corrente.

L'installazione la fai tu, ad esempio:

```bash
uv pip install ~/wheels/vllm-0.30.0+cu129-....whl ~/wheels/torch-2.13.0+cu129-....whl
# oppure lasciando risolvere le dipendenze a uv:
uv pip install vllm==0.30.0 --extra-index-url https://wheels.vllm.ai/0.30.0/cu130
```

Con `get vllm` vengono scaricate entrambe le wheel (vllm + torch pinnata); se
la versione di torch richiesta non esiste per nessuna variante compatibile,
vllm viene comunque scaricata e la nota appare su stderr.

## Come sceglie

1. Trova la release piu' recente (GitHub releases per vllm, PyPI per torch).
2. Legge le varianti CUDA disponibili (`cu126`, `cu128`, `cu130`, ...).
3. Sceglie la variante piu' alta il cui major CUDA e' `<=` al major del driver
   (es. driver CUDA 12.6 -> `cu129` se disponibile, altrimenti `cu128`, ...).
4. Filtra le wheel compatibili con il Python e la piattaforma correnti e
   preferisce il tag esatto (`cp312`) rispetto a `abi3`/`py3`.

Se la release piu' recente non ha nessuna variante compatibile, `wheelget`
ripiega sulla release piu' recente che ne ha una e lo segnala su stderr.

Opzioni principali: `--cuda`, `--variant`, `--python`, `--refresh`, `-q`,
`-o/--output`, `-f/--force`. Vedi `wheelget <comando> --help`.

## Note

- Le pagine degli indici vengono messe in cache in `~/.cache/wheelget`
  (usa `--refresh` per ignorarla).
- Per evitare il rate limit dell'API GitHub puoi impostare `GITHUB_TOKEN`.
- `get` scarica solo il file `.whl` (con verifica dello sha256 quando l'indice
  lo fornisce); non installa nulla.
