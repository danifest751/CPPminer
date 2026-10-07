# Набор для сервера с AMD Radeon AI PRO R9700 (RDNA4, gfx1201)

Pearl и Quantus на OpenCL, по процессу майнера на каждую карту. Рассчитан на Ubuntu 22.04/24.04
с драйвером amdgpu.

## Что внутри

| Файл | Назначение |
|---|---|
| `bin/` | `cppminer` (OpenCL + CPU), ядра `kernels/*.cl`, `libgomp.so.1` |
| `config.env` | кошелёк, имя воркера, пулы, номера карт, доп. переменные |
| `setup.sh` | проверяет драйвер и OpenCL; если карты не видны, ставит OpenCL-рантайм ROCm |
| `check.sh` | проверки корректности на каждой карте (Pearl и Quantus) |
| `bench.sh` | замер скорости всех вариантов ядер, потом обеих карт сразу |
| `mine.sh pearl` / `mine.sh quantus` | майнинг на всех картах в фоне с автоперезапуском |
| `status.sh` | скорость, шары, перезапуски, телеметрия карт |
| `stop.sh` | остановить майнинг |
| `collect.sh` | упаковать логи и сведения о системе в архив, чтобы прислать |
| `cppminer-src.tar` | исходники на случай, если бинарник придётся собрать на месте |

## Порядок

```bash
tar xzf r9700-kit.tar.gz && cd r9700-kit
sudo ./setup.sh          # 1. драйвер/OpenCL, список карт
./check.sh               # 2. корректность (~10-30 мин): должно быть "all passed"
./bench.sh               # 3. скорость (~15 мин): сводка в logs/bench-*/summary.txt
./mine.sh pearl          # 4. майнинг (или ./mine.sh quantus)
./status.sh              #    смотреть, как идёт
./collect.sh             # 5. архив логов для разбора
```

## Что проверяет check.sh

- **Pearl, `--align-test`:** слова вех GEMM с карты сравниваются с CPU. Перед этим запускается
  самотест WMMA. На gfx12 он проверяет обе возможные раскладки операндов и берёт ту, что прошла.
  В логе ищите строки `WMMA self-test gfx12 ... PASS`. Если WMMA не прошёл, майнер сам уходит на
  sudot4 (медленнее, но корректно).
- **Pearl, mock:** найденная шара проверяется полным zk-pow (`verify OK`).
- **Quantus, mock:** самотест ядра при старте, автоподбор варианта, затем шара проверяется
  эталонным хэшем.

## Что смотреть в bench.sh

Для каждой карты: Pearl `wmma` (по умолчанию), `wmma-nopipe` и `sudot4` в TMAC/s, а также
Quantus в MH/s. Если `wmma-nopipe` быстрее, перед майнингом задайте
`PEARL_ENV="CP_OCL_WMMA_PIPELINE=0"` в `config.env`.

## Если что-то не так

- `setup.sh` не видит карт: `dmesg | grep -i amdgpu`, пользователь должен быть в группах `render`
  и `video` (перелогиниться после `setup.sh`).
- Майнер пишет `another cppminer is already mining on OpenCL device N`: на карте уже есть
  процесс, `./stop.sh`.
- Бинарник не запускается (библиотеки): собрать из исходников:
  `mkdir src && tar xf cppminer-src.tar -C src && cd src && ./build.sh --backend cpu,opencl`.
- Любые сбои: `./collect.sh` и прислать архив `logs/r9700-report-*.tar.gz`.

## Переменные для экспериментов

- `CP_OCL_WMMA_PIPELINE=0`: WMMA без двойной буферизации операндов.
- `CP_OCL_WMMA_G12_KSPLIT=0|1`: принудительная раскладка WMMA на gfx12 (без автоподбора).
- `--ocl-dot sudot`: Pearl без матричных ядер (int8 dot4).
- `CP_QPOW_OCL_MUL=1|3`: вариант умножения в Quantus вместо автоподбора.
