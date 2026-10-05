# Hệ thống khuyến nghị cổ phiếu cá nhân hóa

Project kết hợp **tín hiệu nghiên cứu cổ phiếu** với **hành vi giao dịch lịch sử của
khách hàng** để tạo khuyến nghị phù hợp cho từng người tại thời điểm `t`.

Hệ thống không tự thay thế bộ phận research để dự đoán cổ phiếu nào sinh lời. Research
quyết định mã nào đang có tín hiệu `BUY`, `HOLD`, `SELL`; model hành vi chỉ xếp mức độ
phù hợp của các tín hiệu đó với từng khách hàng.

## 1. Bài toán cần giải quyết

Tại mỗi thời điểm `t`, hệ thống trả lời hai câu hỏi:

1. Trong các mã research đang đánh giá `BUY`, mã nào phù hợp nhất với khách hàng?
2. Với các mã khách đang sở hữu, mã nào nên `BÁN`, `GIỮ` hoặc `CÓ THỂ MUA THÊM`?

Hai nhánh được tách riêng:

| Nhánh | Candidate | Kết quả |
|---|---|---|
| Mua mới | Chỉ mã research có `recommendation = BUY` | Top-K mã phù hợp nhất |
| Danh mục | Chỉ mã khách đang sở hữu | BÁN, GIỮ, CÓ THỂ MUA THÊM hoặc CHƯA CÓ TÍN HIỆU |

Nguyên tắc quan trọng:

```text
Research quyết định chất lượng/tín hiệu đầu tư.
Model hành vi quyết định mức độ phù hợp với từng khách.
```

## 2. Kiến trúc tổng thể

```text
7 file CSV
   │
   ▼
[S1] ETL + đặc trưng thị trường trong DuckDB
   │
   ▼
[S2] Trạng thái khách hàng point-in-time
   │
   ▼
[S3] Neo4j: Customer – TRADED – Stock – Sector
   │
   ▼
[S4] Candidate research BUY + vị thế danh mục
   │
   ▼
[S5] LightGBM hành vi BUY/SELL + walk-forward
   │
   ▼
Gradio: TOP-K mua mới + hành động cho danh mục
```

- **DuckDB** lưu dữ liệu đã làm sạch, feature và bảng huấn luyện.
- **Neo4j** lưu quan hệ khách–cổ phiếu–ngành và tạo đặc trưng graph.
- **LightGBM** học hành vi khách hàng từ các feature dạng bảng.
- **Gradio** hiển thị kết quả khuyến nghị.

LightGBM không chạy trực tiếp bên trong Neo4j. Neo4j tạo các con số như độ đồng mua,
số người mua/bán và giá trị giao dịch; sau đó các cột này được đưa vào LightGBM.

## 3. Dữ liệu đầu vào

Thư mục `data/` gồm:

| File | Vai trò |
|---|---|
| `securities_master.csv` | Danh mục mã, sàn và ngành ICB |
| `stock_market_daily.csv` | Giá và khối lượng cổ phiếu theo ngày |
| `market_index_daily.csv` | Dữ liệu VNINDEX và các chỉ số |
| `customer_profile.csv` | Hồ sơ, khẩu vị rủi ro, kỳ hạn khách hàng |
| `customer_transactions_raw.csv` | Lịch sử BUY/SELL của khách |
| `customer_holdings.csv` | Snapshot danh mục tại các mốc cuối tháng |
| `candidate_stocks_research.csv` | Tín hiệu `BUY/HOLD/SELL`, MarketScore và thứ hạng |

Các cột quan trọng của bảng research:

| Cột | Ý nghĩa |
|---|---|
| `research_date` | Thời điểm tín hiệu có hiệu lực |
| `stock_code` | Mã cổ phiếu |
| `recommendation` | `BUY`, `HOLD` hoặc `SELL` |
| `market_score` | Điểm chất lượng/hấp dẫn do research cung cấp |
| `candidate_rank` | Thứ hạng của mã trong danh sách research |

## 4. MarketScore được dùng như thế nào?

### 4.1 MarketScore không được project tự tính

`market_score` được đọc trực tiếp từ `candidate_stocks_research.csv`. Project không tự
tạo điểm này từ giá, momentum hoặc LightGBM. Vì vậy chất lượng sinh lời của kết quả cuối
phụ thuộc vào chất lượng tín hiệu research đầu vào.

### 4.2 Lọc candidate

Nhánh mua mới chỉ nhận mã thỏa mãn:

```text
recommendation(t, stock) = BUY
```

Nếu một mã không nằm trong danh sách `BUY` tại `t`, model hành vi không được phép đưa mã
đó vào danh sách mua, dù khách từng giao dịch mã đó nhiều lần.

Nếu số lượng mã BUY vượt `N_CANDIDATES`, hệ thống ưu tiên theo `candidate_rank` rồi lấy
tối đa `N_CANDIDATES` mã.

### 4.3 Chuẩn hóa MarketScore

MarketScore được chuẩn hóa Min-Max riêng tại từng thời điểm:

```text
MarketScoreNorm(s, t)
    = (MarketScore(s, t) - MinScore(t))
      / (MaxScore(t) - MinScore(t))
```

Kết quả nằm trong `[0, 1]`. Nếu toàn bộ mã tại `t` có cùng điểm, mẫu số bằng 0 và code
gán `MarketScoreNorm = 0` để tránh lỗi chia cho 0.

### 4.4 Công thức điểm mua cuối

LightGBM trả về xác suất hành vi mua. Xác suất này tiếp tục được chuẩn hóa Min-Max trong
từng nhóm `khách hàng × thời điểm`:

```text
BehaviorScoreNorm(c, s, t)
    = (P_buy(c, s, t) - MinP(c, t))
      / (MaxP(c, t) - MinP(c, t))
```

Điểm xếp hạng cuối:

```text
FinalScore(c, s, t)
    = MARKET_WEIGHT × MarketScoreNorm(s, t)
      + (1 - MARKET_WEIGHT) × BehaviorScoreNorm(c, s, t)
```

Mặc định `MARKET_WEIGHT = 0.60`:

```text
FinalScore = 60% MarketScoreNorm + 40% BehaviorScoreNorm
```

### 4.5 Ví dụ tính điểm

Giả sử research có ba mã BUY:

| Mã | MarketScore | MarketScoreNorm | Xác suất model | BehaviorScoreNorm |
|---|---:|---:|---:|---:|
| FPT | 85 | 1,00 | 0,32 | 0,00 |
| MBB | 75 | 0,50 | 0,70 | 1,00 |
| HPG | 65 | 0,00 | 0,55 | 0,61 |

Với trọng số 60/40:

```text
FPT = 0,60 × 1,00 + 0,40 × 0,00 = 0,600
MBB = 0,60 × 0,50 + 0,40 × 1,00 = 0,700
HPG = 0,60 × 0,00 + 0,40 × 0,61 = 0,244
```

Thứ hạng cuối là `MBB → FPT → HPG`. MBB có MarketScore thấp hơn FPT nhưng phù hợp với
hành vi khách hơn, nên được đưa lên đầu. Tuy nhiên HPG vẫn chỉ được xét vì research đã
đánh giá HPG là `BUY`.

## 5. Các đặc trưng thị trường

Tại bước `s1_etl.py`, hệ thống tính:

### Momentum

```text
ret_N(t) = Price(t) / Price(t-N) - 1
```

Gồm:

- `ret_5`: lợi suất 5 phiên;
- `ret_20`: lợi suất 20 phiên;
- `ret_60`: lợi suất 60 phiên.

### Biến động 20 phiên

```text
r1(t)   = Price(t) / Price(t-1) - 1
vol_20  = StdDev(r1 trong 20 phiên gần nhất)
```

### Tỷ lệ phiên tăng

```text
up_ratio_20 = số phiên có r1 > 0 / số phiên quan sát trong cửa sổ 20 phiên
```

### Khối lượng và thanh khoản

```text
vol_ratio = Volume(t) / AverageVolume20(t)
turnover  = Volume(t) × Price(t)
```

Project cũng tính `fwd_ret`, `idx_fwd_ret` và `excess_fwd` để audit kết quả trong tương
lai. Các cột tương lai này bị chặn khỏi feature của LightGBM, không được dùng khi dự đoán.

## 6. Hệ thống học sở thích khách hàng thế nào?

Mọi feature hành vi tại `t` chỉ dùng giao dịch có ngày `< t`.

### 6.1 Trọng số giảm theo thời gian

Lệnh mua càng gần hiện tại có trọng số càng lớn:

```text
w = transaction_value × 0,5 ^ (days_ago / BEHAVIOR_HALF_LIFE)
```

Mặc định `BEHAVIOR_HALF_LIFE = 90` ngày. Một lệnh cách 90 ngày còn một nửa trọng số;
một lệnh cách 180 ngày còn một phần tư trọng số.

Từ đó tính phong cách trung bình có trọng số của khách:

```text
preferred_ret_20       = Σ(ret_20 × w) / Σw
preferred_vol_20       = Σ(vol_20 × w) / Σw
preferred_log_turnover = Σ(log(turnover) × w) / Σw
```

### 6.2 Mức yêu thích một mã trong lịch sử

```text
HistoricalPreference
  = 40% × (giá trị mua mã / tổng giá trị mua của khách)
  + 30% × (số lần mua mã / tổng số lần mua của khách)
  + 30% × 0,5 ^ (số ngày từ lần mua gần nhất / half-life)
```

Điểm cao khi khách từng mua mã với giá trị lớn, mua nhiều lần hoặc vừa mua gần đây.

### 6.3 Mức phù hợp phong cách

```text
SuitabilityScore = (
    SectorShare
    + exp(-|vol_20 - preferred_vol_20| / 0,03)
    + exp(-|ret_20 - preferred_ret_20| / 0,15)
    + exp(-|log_turnover - preferred_log_turnover| / 5)
) / 4
```

Điểm cao khi mã hiện tại gần với ngành, mức biến động, momentum và thanh khoản mà khách
thường lựa chọn.

Ngoài ra model còn nhận:

- số lệnh BUY/SELL và tổng giá trị giao dịch;
- số mã từng giao dịch;
- lần giao dịch/mua/bán gần nhất;
- khách đang nắm mã hay chưa;
- số vị thế, giá trị và lãi/lỗ danh mục;
- loại khách, mức chịu rủi ro, kỳ hạn và tuổi tài khoản.

### 6.4 Đặc trưng hành vi bổ sung

Các đặc trưng dưới đây được ghép vào cả nhánh MUA và nhánh danh mục.
Với dữ liệu upstream, mua theo nghiên cứu nghĩa là mã có khuyến nghị `BUY`
trong `research` đúng ngày giao dịch; không suy diễn thời gian mở khuyến nghị
từ bộ dữ liệu PHS riêng. Chỉ tính kết quả sau khi đủ `HORIZON` phiên tại mốc `t`.
Các cột có hậu tố `90d` dùng `RECENT_WINDOW` (mặc định 90 ngày).

| Đặc trưng | Ý nghĩa |
|---|---|
| `phs_hit_rate`, `phs_avg_excess`, `phs_recent_hit_rate` | Chất lượng các lần khách từng mua theo PHS, chỉ tính lệnh đã đủ `HORIZON` trước `t` |
| `icb_share_30d`, `icb_share_90d`, `icb_recent_shift_30d`, `icb_recent_shift_90d` | Khẩu vị ngành gần đây so với toàn lịch sử mua của khách |
| `stock_in_last_3_buys`, `stock_in_last_5_buys` | Mã đang xét có nằm trong các lệnh BUY gần nhất của khách không |
| `same_sector_last_3_buys`, `same_sector_last_5_buys` | Ngành của mã đang xét có trùng với chuỗi BUY gần nhất của khách không |
| `days_since_last_buy_same_sector` | Số ngày từ lần gần nhất khách mua một mã cùng ngành |

## 7. Đặc trưng graph từ Neo4j

### Node và edge

- `Customer`: khách hàng;
- `Stock`: cổ phiếu;
- `Sector`: ngành ICB;
- `Customer -[:TRADED]-> Stock`: giao dịch có `d`, `side`, `qty`, `value`;
- `Stock -[:IN_SECTOR]-> Sector`: quan hệ mã–ngành.

Trong cửa sổ `COOC_WINDOW` ngày trước `t`, hệ thống tính:

- `cooc_score`: số khách cùng mua hai mã;
- `n_buyers`, `buy_value`: độ phổ biến phía mua;
- `n_sellers`, `sell_value`: độ phổ biến phía bán.

Các điểm được chuẩn hóa và ghép:

```text
GraphScore = 75% × CoocScoreNorm + 25% × BuyerPopularityNorm
```

Một điểm cá nhân hóa sơ bộ cũng được tạo làm feature cho model:

```text
PersonalizationScore
    = 50% × GraphScore
      + 30% × SuitabilityScore
      + 20% × HistoricalPreference
```

Đây chưa phải `BehaviorScore` cuối. `BehaviorScore` cuối là xác suất do LightGBM học từ
PersonalizationScore cùng toàn bộ feature hành vi, thị trường, graph và hồ sơ khách.

## 8. Model BUY học gì?

Candidate huấn luyện là từng bộ `(t, customer, research BUY stock)`.

Nhãn:

```text
y_buy = 1 nếu khách mua mã đó trong HORIZON phiên sau t
y_buy = 0 nếu không mua
```

Mặc định `HORIZON = 20` phiên. Các cột `market_score`, `candidate_rank` và
`market_score_norm` bị loại khỏi model hành vi để tránh model trộn tín hiệu research với
sở thích khách. Chúng chỉ được ghép lại sau khi LightGBM dự đoán.

Do nhãn mua khá thưa, LightGBM cân bằng lớp bằng:

```text
scale_pos_weight = min(number_of_negative / number_of_positive, 50)
```

## 9. Khuyến nghị BÁN/GIỮ/MUA THÊM

Nhánh danh mục chỉ xét mã khách đang sở hữu tại `t`. Action do research quyết định:

| Research | Action hiển thị | Điểm tầng |
|---|---|---:|
| `SELL` | `BÁN` | 1,0 |
| `HOLD` | `GIỮ` | 0,5 |
| `BUY` | `CÓ THỂ MUA THÊM` | 0,0 |
| Không có research | `CHƯA CÓ TÍN HIỆU` | -0,5 |

Model portfolio học:

```text
y_sell = 1 nếu khách bán mã đang nắm trong HORIZON phiên sau t
```

Feature gồm lịch sử BUY/SELL, thời gian giữ, lãi/lỗ, tỷ trọng vị thế, số vị thế, hồ sơ
khách và trạng thái thị trường. Tín hiệu research và MarketScore bị loại khỏi model hành
vi bán.

Thứ tự trên giao diện luôn là:

```text
SELL > HOLD > BUY > chưa có tín hiệu
```

Trong cùng một tầng, `SellBehaviorScore` được dùng để sắp mã nào cần xem trước. Model
không được phép đổi một tín hiệu `SELL` thành `BUY`, hoặc ngược lại.

## 10. Chống rò rỉ dữ liệu tương lai

Các biện pháp đang áp dụng:

1. Feature hành vi chỉ dùng giao dịch có `transaction_date < t`.
2. Truy vấn Neo4j cũng dùng `r.d < t`.
3. Không sử dụng `portfolio_value` và `cash_balance` cuối kỳ làm feature quá khứ.
4. Tay nghề lịch sử chỉ dùng lệnh đã hoàn tất đủ 20 phiên trước `t`.
5. `fwd_ret`, `excess_fwd` và nhãn tương lai không nằm trong feature model.
6. Mốc cuối dữ liệu chưa đủ `HORIZON` có `label_complete = 0` và bị loại khỏi train/test.
7. Đánh giá theo thời gian, không chia train/test ngẫu nhiên.

Những mốc `label_complete = 0` vẫn được giữ để app chấm điểm hiện tại; chúng chỉ không
được dùng làm nhãn huấn luyện.

## 11. Đánh giá walk-forward

Model được đánh giá mở rộng theo quý:

```text
Quý 1..4 → test quý 5
Quý 1..5 → test quý 6
Quý 1..6 → test quý 7
...
```

Các metric chính:

- `AUC`: khả năng phân biệt hành động có/không;
- `Precision@K`: tỷ lệ mã trong Top-K thực sự được khách hành động;
- `Recall@K`: tỷ lệ hành động thực tế được Top-K bắt được;
- `HitRate@K`: tỷ lệ khách/ngày có ít nhất một mã đúng trong Top-K;
- `NDCG@K`: chất lượng thứ tự, mã đúng ở vị trí cao được thưởng nhiều hơn.

Kết quả kiểm thử tích hợp gần nhất:

| Nhánh | AUC | NDCG@10 | Recall@10 | HitRate@10 |
|---|---:|---:|---:|---:|
| BUY | 0,7264 | 0,3356 | 0,5803 | 0,6918 |
| Danh mục | 0,7525 | 0,7526 | 0,9518 | 0,9911 |

Baseline BUY chỉ dùng MarketScore có NDCG@10 là `0,3051`; kết hợp MarketScore với hành
vi đạt `0,3356`. Đây là metric phù hợp hành vi, không phải bằng chứng đảm bảo lợi nhuận.

## 12. Pipeline theo từng file

| Bước | File | Chức năng chính |
|---|---|---|
| S1 | `src/s1_etl.py` | CSV → DuckDB, chuẩn hóa giá, tính market feature |
| S2 | `src/s2_state.py` | Dựng trạng thái và sở thích khách tại từng `t` |
| S3 | `src/s3_graph.py` | Nạp Neo4j, tính đồng mua và độ phổ biến BUY/SELL |
| S4 | `src/s4_features.py` | Hard-filter research BUY, tạo bảng BUY và portfolio |
| S5 | `src/s5_train.py` | Huấn luyện LightGBM và đánh giá walk-forward |
| App | `src/app.py` | Ghép điểm, hiển thị khuyến nghị bằng Gradio |
| Orchestrator | `src/pipeline.py` | Chạy tuần tự S1 → S5 |



## 13. Cách chạy bằng Docker

### Yêu cầu

- Docker Desktop;
- nên cấp ít nhất 8 GB RAM cho Docker;
- đủ 7 file CSV trong thư mục `data/`.

### Tạo `.env`

Nếu chưa có `.env`:

```cmd
copy .env.example .env
```

Đặt cùng một mật khẩu Neo4j ở hai biến:

```env
NEO4J_AUTH=neo4j/mat_khau_cua_ban
NEO4J_PASSWORD=mat_khau_cua_ban
```

Mật khẩu phải có ít nhất 8 ký tự. `.env` đã được `.gitignore`, không được push lên Git.

### Khởi động

```cmd
cd /d "D:\RCM-STOCK\data\New folder"
docker compose up -d --build
docker compose logs -f app
```

Lần đầu pipeline phải ETL, nạp graph và huấn luyện nhiều fold nên có thể mất khoảng
15–60 phút tùy cấu hình máy. Khi log báo Gradio đã khởi động, mở:

- Ứng dụng: <http://localhost:7860>
- Neo4j Browser: <http://localhost:7474>

Nhấn `Ctrl+C` chỉ dừng theo dõi log, container vẫn chạy.

### Kiểm tra trạng thái

```cmd
docker compose ps
docker compose logs --tail 200 app
docker compose logs --tail 100 neo4j
```

### Ép chạy lại toàn bộ pipeline

```cmd
docker compose run --rm app python /app/src/pipeline.py --force
docker compose restart app
```

## 14. Tham số cấu hình

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `HORIZON` | 20 | Số phiên tương lai dùng để tạo nhãn hành vi |
| `N_CANDIDATES` | 50 | Số candidate BUY tối đa mỗi khách/ngày |
| `TOPK` | 10 | Số mã mua mới trả về |
| `COOC_WINDOW` | 180 | Cửa sổ ngày tính quan hệ graph |
| `COOC_TOPN` | 25 | Số quan hệ đồng mua mạnh giữ lại mỗi mã |
| `MARKET_WEIGHT` | 0,60 | Trọng số MarketScore trong điểm mua cuối |
| `BEHAVIOR_HALF_LIFE` | 90 | Chu kỳ bán rã lịch sử giao dịch, đơn vị ngày |
| `DUCKDB_MEMORY` | 3GB | Giới hạn RAM DuckDB trong container |
| `GRADIO_PORT` | 7860 | Cổng giao diện |

Thay đổi feature, `HORIZON`, candidate hoặc trọng số cần chạy lại pipeline để kết quả
đánh giá và model được đồng bộ.

## 15. Cấu trúc repository

```text
data/                    dữ liệu CSV đầu vào
src/                     source code pipeline và ứng dụng
docs/                    phương pháp, khảo sát dữ liệu, kết quả chi tiết
artifacts/               DuckDB/model sinh tự động, không commit
docker-compose.yml       Neo4j + application services
Dockerfile               image chạy Python
requirements.txt         dependency runtime
requirements-dev.txt     công cụ kiểm tra code
pyproject.toml           cấu hình Ruff
.env.example             mẫu cấu hình, không chứa mật khẩu thật
```


## 16. Kiểm tra chất lượng code

```bash
python -m pip install -r requirements-dev.txt
ruff check src
ruff format --check src
docker compose config --quiet
```

## 17. Giới hạn và lưu ý sử dụng

- Model học khả năng phù hợp/hành vi, không bảo đảm cổ phiếu sẽ sinh lời.
- Chất lượng đầu tư phụ thuộc trực tiếp vào bảng research đầu vào.
- Nếu mã đang nắm không có research tại `t`, hệ thống trả `CHƯA CÓ TÍN HIỆU` thay vì
  tự suy diễn BUY/HOLD/SELL.
- Backtest thực tế cần bổ sung phí giao dịch, thuế, trượt giá, thanh khoản và giới hạn
  tỷ trọng theo mã/ngành.
- Kết quả của hệ thống chỉ phục vụ nghiên cứu và hỗ trợ quyết định, không phải cam kết
  lợi nhuận hay tư vấn đầu tư tự động.
