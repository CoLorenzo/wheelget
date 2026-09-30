# wheelget

Trova e scarica la wheel giusta per la tua GPU NVIDIA: legge la versione di CUDA
del driver installato e sceglie la release piu' recente del pacchetto compatibile
con quella CUDA.

- `wheelget url <pkg>` stampa il link diretto della wheel
- `wheelget get <pkg>` scarica la wheel nella directory corrente

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
```

L'installazione la fai tu, ad esempio:

```bash
uv pip install ~/wheels/vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl
# oppure lasciando risolvere le dipendenze a uv:
uv pip install vllm==0.30.0 --extra-index-url https://wheels.vllm.ai/0.30.0/cu130
```

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
