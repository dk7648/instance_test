# 🚀 AWS EC2 Instance Performance 분석 리포트

본 보고서는 **AWS EC2의 다양한 인스턴스 제품군과 아키텍처(x86 vs ARM)** 에 따른 성능 차이를

**실제 측정 데이터(Throughput: rows/s) 기반으로 분석**한 결과입니다.

---

## 🎯 1. 측정 목적

- **인스턴스 최적화**
    
    서비스 워크로드 특성(CPU 연산 중심 vs I/O 중심)에 맞는 최적의 인스턴스 타입 선정
    
- **아키텍처 검증**
    
    Intel/AMD 기반 **x86** 아키텍처와 **Graviton(ARM)** 아키텍처 간 실질 성능 비교
    
- **스토리지 성능 확인**
    
    고밀도 로컬 스토리지(D2), Nitro SSD(i 시리즈), 로컬 NVMe(c6id) 등 스토리지 특성에 따른 효율 비교
    

---

## 📊 2. 측정 지표

- **Throughput (rows/s)**
    - 1초당 처리한 데이터의 행(row) 수
    - 값이 클수록 더 많은 데이터를 빠르게 처리했음을 의미

---

## 🛠️ 3. 측정 방법론

| 테스트 항목 | 설명 |
| --- | --- |
| **CPU_HASH (계산/CPU)** | SHA256 해싱 연산을 통해 CPU의 순수 연산 성능 측정 |
| **IO_BUFFERED_WRITE (파일 쓰기)** | OS 캐시 영향을 포함한 파일 시스템 쓰기 처리량 측정 *(fsync 미사용)* |
| **SQLITE_PRAGMAS (DB 빠른 모드)** | WAL/NORMAL/mmap/cache 등 최적화 설정을 적용한 SQLite 트랜잭션 처리량 측정 | 

---

## 💻 4. 분석 대상 인스턴스 사양

> 모든 인스턴스는 4 vCPU 기준
> 

| 아키텍처 | 인스턴스 타입 | 메모리 | 특징 |
| --- | --- | --- | --- |
| **x86** | `m6i.xlarge` | 16 GiB | 범용 인스턴스 |
| **x86** | `i4i.xlarge` | 32 GiB | 스토리지 최적화 (Nitro SSD) |
| **x86** | `d2.xlarge` | 30.5 GiB | 고밀도 로컬 스토리지(HDD 계열) |
| **x86** | `c6id.xlarge` | 8 GiB | 컴퓨팅 최적화 + 로컬 NVMe |
| **ARM** | `m6g.xlarge` | 16 GiB | 범용 (Graviton2) |
| **ARM** | `i8g.xlarge` | 32 GiB | 최신 Graviton4 + 고성능 스토리지 |
| **ARM** | `c6g.xlarge` | 8 GiB | 컴퓨팅 최적화 (Graviton2) |

---

## 📈 5. 성능 비교 결과

### 5-1. 🆚 x86 인스턴스 간 성능 비교

x86 환경에서는 **`c6id.xlarge`가 전 테스트에서 1등**을 기록했습니다.

| 테스트 항목 (rows/s) | M (m6i) | I (i4i) | D (d2) | C (c6id) | 1등 |
| --- | --- | --- | --- | --- | --- |
| **CPU_HASH** | 39,171 | 39,258 | 28,789 | **39,642** | C |
| **IO_BUFFERED_WRITE** | 2,960,253 | 3,012,751 | 2,419,675 | **3,071,144** | C |
| **SQLITE_PRAGMAS** | 458,570 | 454,075 | 385,815 | **466,799** | C |

<img width="3000" height="1800" alt="benchmark_cpu_hash" src="https://github.com/user-attachments/assets/ec0a3fd1-9831-4218-a120-48d8d212e73f" />
<img width="3000" height="1800" alt="benchmark_io_buffered_write" src="https://github.com/user-attachments/assets/06e8cbf3-8c4f-4c71-8f4e-13035e5f6430" />
<img width="3000" height="1800" alt="benchmark_sqlite_pragmas" src="https://github.com/user-attachments/assets/c159d9fa-379d-434a-b3c2-77265ac3133c" />


**결론**

- C/M/I 계열은 전반적으로 **비슷한 성능대**를 형성
- *D(d2)**는 모든 테스트에서 **명확한 성능 저하**가 관찰됨

<details>
<summary><b>💡왜 d2가 낮게 나왔을까?</b></summary>
<div markdown="1">

**1) 💾 스토리지 매체 차이(HDD 성향 vs SSD/NVMe)**

- d2는 “대용량 저장” 목적의 특성이 강해, **저지연 랜덤 I/O**가 중요한 워크로드(파일/DB)에 불리합니다.
- SQLite insert는 내부적으로 잦은 메타데이터/페이지 접근이 발생해 **랜덤 접근 비용**이 성능에 크게 반영됩니다.

**2) 🧱 CPU 세대/마이크로아키텍처 영향**

- `CPU_HASH`는 디스크가 아니라 **순수 연산 + 병렬 효율**을 보는 테스트입니다.
- d2의 낮은 결과는 동일 vCPU라도 **IPC/클럭/명령어 최적화 차이**가 누적된 것으로 해석할 수 있습니다.

**3) 🧩 플랫폼/오프로딩 특성 차이 가능성**

- 최신 계열은 I/O 처리 경로(오프로딩/가상화)에서 이점이 있는 편이고, 구형 계열은 상대적으로 오버헤드가 커질 수 있습니다.

**💡분석 결론: d2의 본래 용도**

- 적합: 대용량 로그/아카이빙/순차 처리 비중이 큰 저장 중심 워크로드
- 부적합: 고성능 연산/저지연 I/O/DB 트랜잭션 중심 워크로드

</div>
</details>

---

### 5-2. 🌍 x86 vs ARM 성능 비교 (전체)

최신 **Graviton4 기반 `i8g.xlarge`** 가 연산 및 I/O 전 영역에서 **최고 성능**을 기록했습니다.

| 테스트 항목 (rows/s) | x86 (c6id) | ARM-m (m6g) | ARM-i (i8g) | ARM-c (c6g) |
| --- | --- | --- | --- | --- |
| **CPU_HASH** | 39,642 | 52,299 | **99,299** | 52,373 |
| **IO_BUFFERED_WRITE** | 3,071,144 | 2,428,265 | **3,183,217** | 2,205,210 |
| **SQLITE_PRAGMAS** | 466,799 | 290,800 | **470,632** | 292,266 |

**핵심 관찰**

- “ARM이 무조건 빠르다”는 결론은 불가
- 내 데이터에서는 **ARM(i8g)은 전천후 우세**, 반면 **ARM(m6g/c6g)은 CPU_HASH만 우세하고 쓰기/SQLite는 약세**가 뚜렷

---

### 5-3. 패밀리별 x86 vs ARM 상세 비교

### 🆚 M 패밀리 (m6i vs m6g)

**내 결과**

- CPU_HASH: ARM(m6g) **+33.5%** 우세
- IO_BUFFERED_WRITE: ARM(m6g) **18.0%** 열세
- SQLITE_PRAGMAS: ARM(m6g) **36.6%** 열세

**결론**

- **CPU 연산 비중이 크면 m6g 고려 가능**
- **쓰기/SQLite 중심이면 m6i가 더 안정적**

---

### 🆚 I 패밀리 (i4i vs i8g)

**내 결과**

- CPU_HASH: ARM(i8g) **+152.9% (2.53배)** 우세
- IO_BUFFERED_WRITE: ARM(i8g) **+5.7%** 우세
- SQLITE_PRAGMAS: ARM(i8g) **+3.6%** 우세

**결론**

- **대량 처리(배치/ETL/데이터 적재)에서는 i8g가 최강의 선택지**
- i4i는 x86 생태계 유지가 필요할 때 **실용적인 상위권 옵션**

---

### 🆚 C 패밀리 (c6id vs c6g)

**내 결과**

- CPU_HASH: ARM(c6g) **+32.1%** 우세
- IO_BUFFERED_WRITE: ARM(c6g) **28.2%** 열세
- SQLITE_PRAGMAS: ARM(c6g) **37.4%** 열세

**결론**

- **DB 적재/쓰기 중심 → c6id**
- **순수 CPU 연산 중심 → c6g 고려 가능**

---

## 🧐 6. 분석 및 종합 결론

1. **Graviton4(i8g)의 강력한 개선**
    - i8g는 단순 가성비를 넘어 **CPU + I/O + SQLite 모두 상위권**
2. **쓰기/DB 적재는 인스턴스 조합에 따라 성향이 크게 갈림**
    - c6id처럼 (구성상) 쓰기/DB 경로가 강한 타입이 **확실히 우세**
3. **d2는 “대용량 저장” 목적에 최적**
    - 본 테스트처럼 **연산/저지연/DB 트랜잭션** 중심에는 부적합

---

## 🏆 최종 요약

- **최고 성능 지향(전천후)**
    - → **`i8g.xlarge` (ARM, Graviton4)**
- **x86 환경 유지 시 최우선**
    - → **`c6id.xlarge`**
- **주의 사항**
    - → **`d2.xlarge`** 는 **대용량 저장** 목적에 적합 (고성능 워크로드에는 비추천)

| 워크로드 성격 | 추천 인스턴스 | 이유 |
| --- | --- | --- |
| **최고의 올라운더 성능** | **i8g.xlarge (ARM)** | 연산 성능 + I/O + SQLite 모두 최상위 |
| **강력한 DB 적재 성능** | **c6id.xlarge (x86)** | IO/SQLite(특히 PRAGMA) 최상위권 |
| **안정적인 범용 서비스** | **m6i.xlarge (x86)** | 범용 밸런스 + 쓰기/SQLite에 안정적 |
| **가성비 연산 중심** | **c6g.xlarge (ARM)** | CPU_HASH 우세(연산 중심 워크로드에 적합) |
| **대용량 저장 중심** | **d2.xlarge (x86)** | 저장 목적에 특화(성능 목적엔 비권장) |

---
<details>
<summary>코드</summary>
<div markdown="1">

```python
"""
EC2 Instance Family Micro-Benchmark (rows/s)

목적
- 인스턴스 패밀리/아키텍처(x86 vs ARM)에 따른 성능 차이를 비교
- CPU 연산(해시), 파일 쓰기(버퍼드/선택적 fsync), SQLite insert(PRAGMA 튜닝) 측정

출력
- 각 테스트별 median/p95/min/max 시간 + throughput(rows/s)
"""

import csv
import hashlib
import os
import platform
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional

# ==========================================================
# 0) SETTINGS
# ==========================================================
CSV_PATH = "sap500_data.csv"
OUTDIR   = "."

LIMIT_ROWS = 0
REPEATS = 5                   # 반복 횟수
WORKERS = "auto"

# --- CPU 테스트 강도 (해시 반복 횟수)
CPU_ROUNDS = 30

# --- 파일 쓰기 테스트 설정
BLOCK_LINES = 5000            # buffered write block 크기
FSYNC_LINES = 0               

# --- SQLite insert 테스트 설정
SQLITE_COMMIT_EVERY = 1000    # 커밋 단위
# ==========================================================

# ==========================================================
# 1) 유틸리티 (시간/통계/시스템 정보)
# ==========================================================
def now_perf():
    return time.perf_counter()

def fmt_sec(x: float) -> str:
    return f"{x:.4f}s"

def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def sys_info():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
    }

def summarize_times(times: List[float]) -> str:
    med = statistics.median(times)
    p95 = percentile(times, 0.95)
    mn = min(times)
    mx = max(times)
    return f"median={fmt_sec(med)}, p95={fmt_sec(p95)}, min={fmt_sec(mn)}, max={fmt_sec(mx)}"
# ==========================================================

# ==========================================================
# 2) 데이터 로딩 (모든 테스트가 동일 데이터 사용)
# ==========================================================
@dataclass
class CsvData:
    header: List[str]
    rows: List[List[str]]

def load_csv_data(file_path: str, limit: Optional[int] = None) -> CsvData:
    rows: List[List[str]] = []
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, r in enumerate(reader):
            rows.append(r)
            if limit is not None and (i + 1) >= limit:
                break
    return CsvData(header=header, rows=rows)

def build_csv_lines_bytes(data: CsvData) -> List[bytes]:
    lines: List[bytes] = []
    lines.append((",".join(data.header) + "\n").encode("utf-8"))
    for r in data.rows:
        lines.append((",".join(r) + "\n").encode("utf-8"))
    return lines
# ==========================================================

# ==========================================================
# 3) TEST 1 : CPU-bound
#    - multiprocessing으로 코어를 최대한 활용
# ==========================================================
def _hash_worker(args):
    chunk_rows, rounds = args
    out = 0
    for _ in range(rounds):
        for r in chunk_rows:
            h = hashlib.sha256()
            for v in r:
                h.update(v.encode("utf-8", errors="ignore"))
            out ^= int.from_bytes(h.digest()[:8], "little")
    return out

def bench_cpu_hash(rows: List[List[str]], rounds: int, workers: int) -> float:
    n = len(rows)
    w = max(1, workers)
    chunk_size = (n + w - 1) // w
    chunks = [rows[i:i + chunk_size] for i in range(0, n, chunk_size)]

    start = now_perf()
    if w == 1:
        _hash_worker((rows, rounds))
    else:
        with Pool(processes=w) as pool:
            pool.map(_hash_worker, [(c, rounds) for c in chunks])
    return now_perf() - start
# ==========================================================

# ==========================================================
# 4) TEST 2: I/O write
# ==========================================================
def bench_io_write(lines: List[bytes], out_path: str, fsync_every: int, block_lines: int) -> float:
    start = now_perf()
    with open(out_path, "wb", buffering=1024 * 1024) as f:
        buf: List[bytes] = []
        for i, line in enumerate(lines, start=1):
            buf.append(line)
            if len(buf) >= block_lines:
                f.write(b"".join(buf))
                buf.clear()

            if fsync_every > 0 and (i % fsync_every == 0):
                f.flush()
                os.fsync(f.fileno())

        if buf:
            f.write(b"".join(buf))

        f.flush()
        if fsync_every > 0:
            os.fsync(f.fileno())

    return now_perf() - start
# ==========================================================

# ==========================================================
# 5) TEST 3: SQLite insert
# ==========================================================
def bench_sqlite_insert_pragmas(
    rows: List[List[str]],
    header: List[str],
    db_path: str,
    commit_every: int,
) -> float:
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-200000;")   # ~200MB
    cur.execute("PRAGMA mmap_size=268435456;")  # 256MB

    # 테이블 생성 (CSV 컬럼을 전부 TEXT로)
    col_def = ", ".join([f'"{c}" TEXT' for c in header])
    cur.execute(f"CREATE TABLE test_table ({col_def})")

    # INSERT 준비
    placeholders = ",".join(["?"] * len(header))
    q = f"INSERT INTO test_table VALUES ({placeholders})"

    # INSERT + 주기적 commit
    start = now_perf()
    pending = 0
    cur.execute("BEGIN;")
    for r in rows:
        cur.execute(q, r)
        pending += 1
        if commit_every > 0 and pending >= commit_every:
            conn.commit()
            cur.execute("BEGIN;")
            pending = 0

    conn.commit()
    conn.close()
    return now_perf() - start
# ==========================================================

# ==========================================================
# 6) Main
# ==========================================================
def main():
    # (A) 입력/출력 경로 확인
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV_PATH not found: {CSV_PATH} (경로를 하드코딩 값으로 수정하세요)")

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # (B) 실행 파라미터 결정
    workers = cpu_count() if WORKERS == "auto" else max(1, int(WORKERS))
    limit = None if LIMIT_ROWS == 0 else LIMIT_ROWS

    # (C) 헤더 출력(재현성 기록)
    print("=" * 72)
    print("System:", sys_info())
    print("CSV_PATH:", CSV_PATH)
    print("OUTDIR:", str(outdir.resolve()))
    print(f"workers={workers}, repeats={REPEATS}, cpu_rounds={CPU_ROUNDS}")
    print(f"block_lines={BLOCK_LINES}, fsync_lines={FSYNC_LINES}, sqlite_commit_every={SQLITE_COMMIT_EVERY}")
    print("=" * 72)

    # (D) 데이터 로딩 (모든 테스트가 동일 데이터 사용)
    data = load_csv_data(CSV_PATH, limit=limit)
    nrows = len(data.rows)
    print(f"[*] Loaded rows: {nrows:,}, cols: {len(data.header)}")

    # 파일쓰기용 bytes 라인 생성(1회)
    lines = build_csv_lines_bytes(data)

    # ------------------------------------------------------
    # (1) CPU_HASH
    # ------------------------------------------------------
    cpu_times = []
    for _ in range(REPEATS):
        cpu_times.append(bench_cpu_hash(data.rows, rounds=CPU_ROUNDS, workers=workers))
    cpu_thr = nrows / statistics.median(cpu_times)

    # ------------------------------------------------------
    # (2) IO_BUFFERED_WRITE
    # ------------------------------------------------------
    io_buf_times = []
    csv_buf_path = str(outdir / "bench_buffered.csv")
    for _ in range(REPEATS):
        io_buf_times.append(bench_io_write(lines, csv_buf_path, fsync_every=0, block_lines=BLOCK_LINES))
    io_buf_thr = nrows / statistics.median(io_buf_times)

    # ------------------------------------------------------
    # (3) IO_FSYNC_WRITE
    # ------------------------------------------------------
    io_fsync_times = []
    io_fsync_thr = None
    if FSYNC_LINES > 0:
        csv_fsync_path = str(outdir / "bench_fsync.csv")
        for _ in range(REPEATS):
            io_fsync_times.append(bench_io_write(lines, csv_fsync_path, fsync_every=FSYNC_LINES, block_lines=1))
        io_fsync_thr = nrows / statistics.median(io_fsync_times)

    # ------------------------------------------------------
    # (4) SQLITE_PRAGMAS
    # ------------------------------------------------------
    sqlite_pragmas_times = []
    db_pragmas_path = str(outdir / "bench_pragmas.db")
    for _ in range(REPEATS):
        sqlite_pragmas_times.append(
            bench_sqlite_insert_pragmas(data.rows, data.header, db_pragmas_path, commit_every=SQLITE_COMMIT_EVERY)
        )
    sqlite_pragmas_thr = nrows / statistics.median(sqlite_pragmas_times)

    # (E) 결과 출력
    print("\n" + "=" * 72)
    print(f"RESULTS (rows={nrows:,})")
    print("=" * 72)

    print(f"\n[CPU_HASH (rounds={CPU_ROUNDS}, workers={workers})]")
    print("  " + summarize_times(cpu_times))
    print(f"  throughput: {cpu_thr:,.2f} rows/s")

    print(f"\n[IO_BUFFERED_WRITE (block_lines={BLOCK_LINES})]")
    print("  " + summarize_times(io_buf_times))
    print(f"  throughput: {io_buf_thr:,.2f} rows/s")

    if FSYNC_LINES > 0:
        print(f"\n[IO_FSYNC_WRITE (fsync_every={FSYNC_LINES} lines)]")
        print("  " + summarize_times(io_fsync_times))
        print(f"  throughput: {io_fsync_thr:,.2f} rows/s")

    print(f"\n[SQLITE_PRAGMAS (commit_every={SQLITE_COMMIT_EVERY})]")
    print("  " + summarize_times(sqlite_pragmas_times))
    print(f"  throughput: {sqlite_pragmas_thr:,.2f} rows/s")

if __name__ == "__main__":
    main()

```

</div>
</details>
