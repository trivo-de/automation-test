# Hệ thống khuyến nghị cổ phiếu cá nhân hóa

Project kết hợp **tín hiệu nghiên cứu cổ phiếu** với **hành vi giao dịch lịch sử của
khách hàng** để tạo khuyến nghị phù hợp cho từng người tại thời điểm `t`.

Hệ thống không tự thay thế bộ phận research để dự đoán cổ phiếu nào sinh lời. Khuyến
nghị của PHS quyết định mã nào vừa có `BUY` mới, mã nào đang `HOLD` do BUY cũ còn mở,
và mã nào vừa được chốt lời/cắt lỗ (`SELL`);
model hành vi chỉ xếp mức độ phù hợp của các tín hiệu đó với từng khách hàng.

## 1. Bài toán cần giải quyết

Tại mỗi mốc quyết định `t` — từng phiên giao dịch (mặc định), phiên cuối mỗi tuần hoặc cuối mỗi tháng, đặt bằng
`DECISION_FREQ` — hệ thống trả lời hai câu hỏi:

1. Trong các mã research vừa đánh giá `BUY` đúng ngày `t`, mã nào phù hợp nhất với khách hàng?
2. Với các mã khách đang sở hữu, mã nào nên `BÁN`, `CÓ THỂ MUA`, `GIỮ` hoặc `CHƯA CÓ TÍN HIỆU`?

Hai nhánh được tách riêng:

| Nhánh | Candidate | Kết quả |
|---|---|---|
| Mua mới | Chỉ mã PHS mới khuyến nghị `BUY` đúng ngày `t` | Top-K mã phù hợp nhất |
| Danh mục | Chỉ mã khách đang sở hữu | BÁN, CÓ THỂ MUA, GIỮ hoặc CHƯA CÓ TÍN HIỆU |

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
- **Neo4j** lưu quan hệ khách–cổ phiếu–ngành, tạo đặc trưng graph và tìm **khách tương tự**
  bằng phép duyệt Khách → Mã ← Khách.
- **LightGBM** học hành vi khách hàng từ các feature dạng bảng.
- **Gradio** hiển thị kết quả khuyến nghị.

LightGBM không chạy trực tiếp bên trong Neo4j. Neo4j tạo các con số như độ đồng mua,
số người mua/bán, giá trị giao dịch và danh sách khách tương tự; sau đó các cột này được
đưa vào LightGBM.

## 3. Dữ liệu đầu vào

Thư mục `data/` gồm:

| File | Vai trò |
|---|---|
| `securities_master.csv` | Danh mục mã, sàn và ngành ICB |
| `stock_market_daily.csv` | Giá và khối lượng cổ phiếu theo ngày |
| `market_index_daily.csv` | Dữ liệu VNINDEX và các chỉ số |
| `customer_profile.csv` | Loại khách và ngày mở tài khoản |
| `customer_transactions_raw.csv` | Lịch sử BUY/SELL của khách |
| `customer_holdings.csv` | Snapshot danh mục cuối tháng — chỉ dùng để đối chiếu (xem dưới) |
| `candidate_stocks_phs_skill.csv` | Nhật ký khuyến nghị PHS: mở `BUY`, đóng bằng `TAKE_PROFIT`/`CUT_LOSS` |

Hồ sơ khách không còn cột khẩu vị rủi ro và kỳ hạn tự khai. Kỳ hạn đầu tư được suy ra
từ hành vi nắm giữ thực tế (mục 6.4).

**Kiểm tra đầu vào.** Trước khi chạy, `src/validate.py` kiểm tra từng file: đủ cột, cột
bắt buộc không trống, số/ngày/giờ đúng định dạng, `side` và `recommendationType` chỉ nhận
giá trị hợp lệ, và có chỉ số `VNINDEX`. Nếu sai, pipeline dừng ngay và in ra file, cột, số
dòng lỗi kèm một giá trị ví dụ.

**Vị thế được dựng lại từ giao dịch.** S2 tính số lượng và giá vốn bình quân (có phí mua
`BUY_FEE` = 0,15%) sau từng lệnh, nên biết được danh mục của khách tại bất kỳ ngày nào chứ
không chỉ cuối tháng. Kết quả được đối chiếu với `customer_holdings.csv`: trên bộ dữ liệu
hiện tại khớp 100% cả số lượng lẫn giá vốn.

Các cột của file khuyến nghị PHS:

| Cột | Ý nghĩa |
|---|---|
| `symbol` | Mã cổ phiếu |
| `recommendationType` | `BUY` (mở khuyến nghị), `TAKE_PROFIT` hoặc `CUT_LOSS` (đóng) |
| `recommendationDate` | Ngày ra khuyến nghị, dạng `dd/mm/yyyy` |
| `recommendationPrice` | Giá hoặc vùng giá (nghìn VND), ví dụ `20.0 - 20.5` |
| `targetPrice`, `cutLossPrice` | Giá mục tiêu và giá cắt lỗ (nghìn VND) |
| `realizedProfitLoss` | Lãi/lỗ thực hiện, chỉ có ở dòng đóng |

S1 ghép lệnh `BUY` thứ k của một mã với lệnh đóng thứ k của mã đó thành bảng
`phs_calls`, rồi dựng tín hiệu tại từng mốc `t`:

| Tín hiệu tại `t` | Điều kiện |
|---|---|
| `BUY` | PHS mở khuyến nghị mua mới đúng ngày `t` |
| `HOLD` | BUY cũ còn mở, nhưng PHS không có tín hiệu mới cho mã đó trong ngày `t` |
| `SELL` | PHS chốt lời/cắt lỗ đúng ngày `t` |
| Không có tín hiệu | Các trường hợp còn lại |

BUY cũ không được lặp lại thành khuyến nghị mua mới ở các ngày sau; nếu khách đang nắm mã đó,
nhánh danh mục sẽ hiển thị `GIỮ` cho tới khi PHS có tín hiệu mới (`BUY`, `TAKE_PROFIT` hoặc
`CUT_LOSS`). Nếu một ngày không có BUY mới, nhánh mua mới không tự gợi ý mã.

## 4. MarketScore được dùng như thế nào?

### 4.1 MarketScore là dư địa tới giá mục tiêu

PHS không cung cấp điểm số sẵn, nên project tính:

```text
MarketScore(s, t) = 100 × (TargetPrice(s) / Close(s, t) - 1)
```

Tức là phần trăm dư địa tăng còn lại từ giá đóng cửa tại `t` tới giá mục tiêu. Công thức
chỉ dùng giá tại `t` nên không rò rỉ tương lai. Chất lượng sinh lời của kết quả cuối vẫn
phụ thuộc vào chất lượng khuyến nghị PHS đầu vào.

### 4.2 Lọc candidate

Nhánh mua mới chỉ nhận mã thỏa mãn:

```text
recommendation(t, stock) = BUY
```

Nếu một mã không nằm trong danh sách `BUY` tại `t`, model hành vi không được phép đưa mã
đó vào danh sách mua, dù khách từng giao dịch mã đó nhiều lần.

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

Các trọng số đặt tay của mục 6.2, 6.3 và 7 được gom về một chỗ trong `src/config.py`
(`HISTORY_PREF_WEIGHTS`, `STYLE_SCALES`, `GRAPH_SCORE_WEIGHTS`, `PERSONALIZATION_WEIGHTS`,
`RESEARCH_SELL_SIGNAL`). Chúng chỉ tạo ra đặc trưng cho model hành vi, không quyết định
trực tiếp điểm cuối.

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
- loại khách và tuổi tài khoản.

### 6.4 Kỳ hạn đầu tư suy ra từ thời gian nắm giữ

S2 tách lịch sử giao dịch thành các **vòng nắm giữ** (`hold_episode`): một vòng bắt đầu
khi khách mua lúc đang không nắm mã đó và kết thúc khi bán hết. Tại mốc `t`, mỗi vòng đã
bắt đầu trước `t` là một bằng chứng:

| Vòng nắm giữ | Bằng chứng |
|---|---|
| Mua → bán hết trong ≤ `SHORT_HOLD_DAYS` (30 ngày) | NGẮN HẠN |
| Giữ 1–3 tháng | Chưa kết luận ở tầng 1, là nhóm "trung hạn" ở tầng 2 |
| Vị thế còn quá mới, hoặc giữ > 90 ngày nhưng đã bán bớt | Chưa kết luận |
| Giữ > `LONG_HOLD_DAYS` (90 ngày) rồi bán hết, hoặc vẫn đang giữ > 90 ngày và chưa bán bớt lần nào | DÀI HẠN |

```text
short_ratio = số bằng chứng ngắn hạn / tổng số bằng chứng
mid_ratio   = số vòng giữ 1–3 tháng  / tổng số bằng chứng
long_ratio  = số bằng chứng dài hạn  / tổng số bằng chứng
```

Mẫu số là toàn bộ bằng chứng, nên các vòng "chưa kết luận" kéo các tỷ lệ xuống. Một mã
được mua bán nhiều vòng thì mỗi vòng là một bằng chứng riêng.

Nhãn được xếp theo hai tầng:

| Tầng | Điều kiện | Nhãn |
|---|---|---|
| 1 — rõ ràng | `short_ratio ≥ HOLD_EVIDENCE_THRESHOLD` | `SHORT` |
| 1 — rõ ràng | `long_ratio ≥ HOLD_EVIDENCE_THRESHOLD` | `LONG` |
| 2 — thiên về | Nhóm ngắn hạn lớn nhất trong ba nhóm ngắn / 1–3 tháng / dài | `LEAN_SHORT` |
| 2 — thiên về | Nhóm 1–3 tháng lớn nhất, hoặc ngắn và dài ngang nhau | `MEDIUM` |
| 2 — thiên về | Nhóm dài hạn lớn nhất | `LEAN_LONG` |
| — | Ít hơn `HOLD_MIN_EVIDENCE` vòng đã kết luận được | `UNRATED` |

Tầng 2 tồn tại vì phần lớn khách không đạt ngưỡng tầng 1 không hề thiếu dữ liệu: họ có
trung vị 73 vòng nắm giữ nhưng hành vi pha trộn. Nếu chỉ dùng tầng 1, 56% khách ở mốc
cuối bị xếp "chưa đánh giá được"; với tầng 2 con số này còn 1,9%.

Nhãn được tính lại tại từng mốc `t` nên một khách có thể đổi nhóm theo thời gian (88% giữ
nguyên nhãn giữa hai mốc cuối tháng liên tiếp). Kiểm tra ngoài mẫu (nhãn tại `t`, hành vi sau `t`):

| Kỳ hạn tại `t` | Số khách ở mốc cuối | Giữ trung vị của vòng mở sau `t` | Bán vị thế trong 20 phiên tới |
|---|---:|---:|---:|
| `SHORT` | 387 | 22 ngày | 75,4% |
| `LEAN_SHORT` | 306 | 40 ngày | 51,7% |
| `MEDIUM` | 478 | 51 ngày | 44,5% |
| `LEAN_LONG` | 302 | 96 ngày | 24,7% |
| `LONG` | 488 | 135 ngày | 15,2% |
| `UNRATED` | 38 | 51 ngày | 36,8% |

### 6.5 Các đặc trưng khách hàng bổ sung

| Đặc trưng | Ý nghĩa |
|---|---|
| `median_hold_days` | Thời gian giữ trung vị của các vòng đã đóng |
| `win_rate`, `avg_realized_ret` | Tỷ lệ vòng có lãi và lãi/lỗ thực hiện trung bình |
| `avg_win_ret`, `avg_loss_ret` | Mức lãi khách thường chốt, mức lỗ khách thường chịu trước khi cắt |
| `risk_appetite_vol` | Biến động trung bình của các mã khách chọn mua, thay cho `risk_level` tự khai |
| `n_buys_recent`, `n_sells_recent`, `n_active_days_recent` | Nhịp giao dịch trong `RECENT_WINDOW` ngày gần nhất |
| `phs_follow_rate` | Tỷ lệ lệnh mua rơi vào lúc PHS đang mở khuyến nghị cho đúng mã đó |
| `phs_hit_rate`, `phs_avg_excess`, `phs_recent_hit_rate` | Chat luong cac lan khach tung mua theo PHS, chi tinh lenh da du `HORIZON` truoc `t` |
| `icb_share_30d`, `icb_share_90d`, `icb_recent_shift_30d`, `icb_recent_shift_90d` | Khau vi nganh gan day so voi toan lich su mua cua khach |
| `stock_in_last_3_buys`, `stock_in_last_5_buys` | Ma dang xet co nam trong cac lenh BUY gan nhat cua khach khong |
| `same_sector_last_3_buys`, `same_sector_last_5_buys` | Nganh cua ma dang xet co trung voi chuoi BUY gan nhat cua khach khong |
| `days_since_last_buy_same_sector` | So ngay tu lan gan nhat khach mua mot ma cung nganh |
| `port_hhi` | Độ tập trung danh mục (1 = dồn vào một mã) |
| `pos_hold_days`, `pos_has_sold` | Vị thế đã giữ bao lâu, đã từng bán bớt chưa |
| `hold_days_vs_typical` | Tuổi vị thế so với thời gian khách thường giữ |
| `pnl_vs_win_habit`, `pnl_vs_loss_habit` | Lãi/lỗ đang treo so với mức khách thường chốt lời/cắt lỗ |

Ba đặc trưng cuối chỉ có ở nhánh danh mục. So với khi bỏ toàn bộ nhóm đặc trưng mới
(3 quý kiểm tra), AUC nhánh danh mục tăng từ 0,760 lên 0,771 và nhánh BUY từ 0,753 lên
0,757.

## 7. Đặc trưng graph từ Neo4j

### Node và edge

- `Customer`: khách hàng;
- `Stock`: cổ phiếu;
- `Sector`: ngành ICB;
- `Customer -[:TRADED]-> Stock`: giao dịch có `d`, `side`, `qty`, `value`;
- `Stock -[:IN_SECTOR]-> Sector`: quan hệ mã–ngành.

Đặc trưng đồ thị được tính tại lát cắt cuối mỗi tháng rồi gán cho mọi mốc quyết định bằng
lát cắt gần nhất `≤ t`. Trong cửa sổ `COOC_WINDOW` ngày trước lát cắt, hệ thống tính:

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

### Khách tương tự

Đây là phần dùng đồ thị đúng nghĩa: một phép duyệt nhiều bước mà bảng phẳng khó diễn đạt.

```text
(Khách A) -[MUA]-> (Mã) <-[MUA]- (Khách B)        bước 1-2: ai giống ai (Neo4j)
(Khách B) -[vừa MUA / BÁN]-> (Mã X)               bước 3: họ vừa làm gì (S4)
```

1. Tại mỗi lát cắt cuối tháng, Neo4j tính độ giống Jaccard giữa mọi cặp khách theo rổ mã
   đã mua trong `COOC_WINDOW` ngày: `số mã mua chung / số mã của cả hai`. Hai khách phải
   có ít nhất `PEER_MIN_SHARED` mã chung; mỗi khách giữ `PEER_TOPK` người giống nhất.
2. S4 nối thêm bước thứ ba: trong `RECENT_WINDOW` ngày ngay trước `t`, các khách tương tự
   đó vừa mua / bán mã nào.

| Đặc trưng | Ý nghĩa |
|---|---|
| `n_peer_buyers`, `peer_buy_score` | Số khách tương tự vừa mua mã này, và tổng độ giống của họ |
| `n_peer_sellers`, `peer_sell_score` | Tương tự cho phía bán |

Khác với đặc trưng cấp khách (kỳ hạn, nhịp giao dịch), các cột này khác nhau giữa từng mã
của cùng một khách, nên giúp xếp hạng bên trong rổ của khách đó. Cả hai nhánh đều dùng,
và app hiển thị số khách tương tự vừa mua / vừa bán bên cạnh từng mã.

### Đồ thị luôn khớp dữ liệu

S3 lưu một **dấu vân tay** (băm nội dung toàn bộ giao dịch, danh mục mã và hồ sơ khách) lên
node `Meta` sau khi nạp xong. Lần chạy sau chỉ bỏ qua bước nạp khi dấu vân tay trùng khớp;
nếu dữ liệu đổi — kể cả khi số dòng giữ nguyên — hoặc lần nạp trước bị ngắt giữa chừng, đồ
thị được xoá theo lô và nạp lại.

Đây chưa phải `BehaviorScore` cuối. `BehaviorScore` cuối là xác suất do LightGBM học từ
PersonalizationScore cùng toàn bộ feature hành vi, thị trường, graph và hồ sơ khách.

## 8. Model BUY học gì?

Candidate huấn luyện là từng bộ `(t, customer, research BUY stock)`.

Nhãn:

```text
y_buy = 1 nếu khách mua mã đó trong HORIZON phiên sau t
y_buy = 0 nếu không mua
```

Mặc định `HORIZON = 20` phiên. Các cột `market_score`, `candidate_rank`,
`market_score_norm`, `target_price` và `cut_loss_price` bị loại khỏi model hành vi để tránh model trộn tín hiệu research với
sở thích khách. Chúng chỉ được ghép lại sau khi LightGBM dự đoán.

Model dùng mục tiêu `binary` và **không** cân bằng lại lớp, nên đầu ra là xác suất thật
(khả năng khách mua mã đó trong `HORIZON` phiên tới) và được hiển thị trực tiếp trong app.

Số vòng boosting không đặt tay: mỗi lần huấn luyện dùng dừng sớm trên quý cuối của tập
train, và model cuối dùng trung vị số vòng của các quý đã đánh giá.

## 9. Khuyến nghị BÁN / MUA / GIỮ

Nhánh danh mục chỉ xét mã khách đang sở hữu tại `t`. Action do research quyết định:

| Tín hiệu PHS tại `t` | Action hiển thị | Điểm tầng |
|---|---|---:|
| `SELL` (vừa chốt lời/cắt lỗ) | `BÁN` | 1,5 |
| `BUY` (mua mới đúng ngày) | `CÓ THỂ MUA` | 0,5 |
| `HOLD` (BUY cũ còn mở, chưa có tín hiệu mới) | `GIỮ` | 0,0 |
| Không có tín hiệu | `CHƯA CÓ TÍN HIỆU` | -0,5 |

Nhật ký PHS không có dòng `HOLD`; hệ thống tự suy ra `HOLD` từ BUY cũ còn hiệu lực.

Model portfolio học:

```text
y_sell = 1 nếu khách bán mã đang nắm trong HORIZON phiên sau t
```

Feature gồm lịch sử BUY/SELL, thời gian giữ, lãi/lỗ, tỷ trọng vị thế, số vị thế, hồ sơ
khách, trạng thái thị trường và hành vi của khách tương tự. Tín hiệu research và
MarketScore bị loại khỏi model hành vi bán.

Vì câu hỏi là "trong danh mục của khách này, mã nào dễ bị bán nhất", model còn nhận vị trí
của từng mã **so với các mã khác trong cùng danh mục**:

| Đặc trưng | Ý nghĩa |
|---|---|
| `pnl_vs_port_avg`, `pnl_rank_in_port` | Lãi/lỗ của mã so với trung bình và thứ hạng trong danh mục |
| `hold_days_rank_in_port` | Mã được giữ lâu hay mới so với các mã còn lại |
| `weight_rank_in_port` | Vị thế lớn hay nhỏ trong danh mục |
| `ret_20_rank_in_port` | Đà giá 20 phiên so với các mã còn lại |
| `recency_rank_in_port` | Mã vừa được giao dịch gần đây hơn các mã còn lại hay không |

Thứ tự trên giao diện luôn là:

```text
SELL > BUY > chưa có tín hiệu
```

Trong cùng một tầng, xác suất bán do model dự đoán được dùng để sắp mã nào cần xem trước. Model
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
8. Có khoảng đệm `HORIZON` phiên giữa train và test: một dòng train chỉ được dùng khi cửa
   sổ nhãn của nó đã kết thúc trước mốc test đầu tiên. Với nhịp tuần, nhãn của các mốc
   liền nhau chồng lên nhau, nên thiếu khoảng đệm này là rò rỉ.
9. Vị thế tại `t` được dựng từ các lệnh tới hết ngày `t`, không dùng snapshot tương lai.

Các nguyên tắc trên được kiểm tra tự động (mục 16): test rò rỉ cắt bỏ toàn bộ dữ liệu sau
một ngày `T`, chạy lại pipeline, và yêu cầu mọi đặc trưng tại các mốc `t ≤ T` giữ nguyên.

Những mốc `label_complete = 0` vẫn được giữ để app chấm điểm hiện tại; chúng chỉ không
được dùng làm nhãn huấn luyện.

## 11. Đánh giá walk-forward

Model được đánh giá mở rộng theo quý, có khoảng đệm `HORIZON` phiên giữa train và test:

```text
Quý 1..4 → test quý 5
Quý 1..5 → test quý 6
Quý 1..6 → test quý 7
...
```

### Cách đo

Mỗi **nhóm** là một cặp (khách, mốc). Thước đo xếp hạng chỉ tính trên nhóm *có thể xếp
sai*: có ít nhất một mã đúng và một mã sai. Nhóm chỉ có một ứng viên, hoặc mọi ứng viên
đều đúng, thì cách xếp nào cũng đạt 1,0 nên bị loại. Số tổng hợp là **gộp mọi nhóm của
mọi quý**, không phải trung bình các quý.

- `HitRate@1`: tỷ lệ nhóm mà mã xếp đầu đúng là mã khách mua/bán;
- `NDCG@3`, `MRR`: chất lượng thứ tự, mã đúng càng ở trên điểm càng cao;
- `Precision@1`: tính trên **mọi** nhóm — mã đứng đầu thực sự được khách hành động;
- `random HitRate@1`: kết quả kỳ vọng nếu chọn ngẫu nhiên, là mốc so sánh thấp nhất;
- `AUC`: khả năng phân biệt hành động có/không trên toàn bộ dòng.

Các mức K cố định là 1 và 3 (`EVAL_KS`), không phụ thuộc `TOPK` hiển thị trong app.

### Kết quả lần chạy gần nhất (nhịp tuần)

| Nhánh | Cách chấm điểm | HitRate@1 | NDCG@3 | MRR | Precision@1 |
|---|---|---:|---:|---:|---:|
| MUA | Chọn ngẫu nhiên | 0,192 | — | — | — |
| MUA | Chỉ MarketScore | 0,192 | 0,409 | 0,438 | 0,013 |
| MUA | Chỉ model hành vi | 0,486 | 0,676 | 0,670 | 0,029 |
| MUA | Điểm cuối (60% MarketScore) | 0,340 | 0,551 | 0,560 | 0,021 |
| Danh mục | Chọn ngẫu nhiên | 0,421 | — | — | — |
| Danh mục | Chỉ tín hiệu PHS | 0,423 | 0,497 | 0,626 | 0,441 |
| Danh mục | Chỉ model hành vi | 0,503 | 0,556 | 0,683 | 0,498 |
| Danh mục | Điểm cuối (PHS trước, hành vi phá hoà) | 0,483 | 0,548 | 0,671 | 0,484 |

- MUA: AUC 0,786, 11 quý test, 9.705 nhóm xếp hạng được trên 175.212 nhóm. Phần lớn nhóm
  không xếp hạng được vì khách không mua mã nào trong rổ PHS: chỉ 1,8% lệnh mua thật của
  khách rơi vào rổ.
- Danh mục: AUC 0,766, 12 quý test, 152.474 nhóm xếp hạng được trên 210.809 nhóm.
- Hai mục tiêu huấn luyện `binary` và `lambdarank` cho kết quả như nhau (MUA 0,486 / 0,485;
  danh mục 0,503 / 0,505), nên giữ `binary` vì nó cho xác suất thật.

### MARKET_WEIGHT đánh đổi cái gì?

Sau mỗi lần chạy, `artifacts/eval_buy_weights.csv` đo lại điểm cuối nhánh MUA ở nhiều mức
`MARKET_WEIGHT` trên cùng một bộ dự đoán ngoài mẫu:

| MARKET_WEIGHT | HitRate@1 | NDCG@3 | Lợi suất vượt trội 20 phiên của mã xếp đầu |
|---:|---:|---:|---:|
| 0,0 (chỉ hành vi) | 0,486 | 0,676 | +2,57% |
| 0,2 | 0,488 | 0,671 | +2,71% |
| 0,4 | 0,474 | 0,647 | +2,83% |
| 0,6 (mặc định) | 0,340 | 0,551 | +3,24% |
| 0,8 | 0,241 | 0,465 | +3,50% |
| 1,0 (chỉ MarketScore) | 0,192 | 0,409 | +3,79% |

Tăng trọng số MarketScore thì mã xếp đầu có lợi suất cao hơn nhưng ít khớp với thứ khách
thực sự mua hơn. Mức 0,4 giữ gần như toàn bộ độ khớp hành vi; từ 0,6 trở lên độ khớp giảm
mạnh.

### MarketScore có dự báo lợi suất không?

`summary.json` có mục `market_score_audit`, đo trên chính rổ PHS đang mở. Mỗi mốc là một
quan sát; sai số chuẩn đã được nới theo số cửa sổ 20 phiên không chồng nhau.

| Phép đo | Số mốc | Trung bình | Sai số chuẩn | t |
|---|---:|---:|---:|---:|
| Cả rổ PHS BUY mới so với VNINDEX | 164 | +2,46% | 0,98% | 2,52 |
| Mã MarketScore cao nhất so với trung bình rổ | 145 | +1,11% | 1,34% | 0,83 |
| Tương quan hạng giữa MarketScore và lợi suất | 123 | +0,080 | 0,098 | 0,81 |

Rổ khuyến nghị của PHS thắng VNINDEX một cách có ý nghĩa thống kê. Việc xếp hạng **bên
trong** rổ bằng MarketScore có chiều hướng đúng nhưng chưa đủ bằng chứng (|t| < 2).

Đây là các thước đo phù hợp hành vi và audit lợi suất quá khứ, không phải bằng chứng đảm
bảo lợi nhuận.

## 12. Pipeline theo từng file

| Bước | File | Chức năng chính |
|---|---|---|
| S1 | `src/s1_etl.py` | CSV → DuckDB, chuẩn hóa giá, mốc quyết định, tín hiệu PHS, market feature |
| Kiểm tra | `src/validate.py` | Kiểm tra cột và kiểu dữ liệu của 7 file đầu vào |
| S2 | `src/s2_state.py` | Dựng lại vị thế từ giao dịch, trạng thái và sở thích khách tại từng `t` |
| S3 | `src/s3_graph.py` | Nạp Neo4j, tính đồng mua, độ phổ biến BUY/SELL và khách tương tự |
| S4 | `src/s4_features.py` | Hard-filter research BUY, tạo bảng BUY và portfolio |
| S5 | `src/s5_train.py` | Huấn luyện LightGBM và đánh giá walk-forward |
| App | `src/app.py` | Ghép điểm, hiển thị khuyến nghị bằng Gradio |
| Orchestrator | `src/pipeline.py` | Kiểm tra đầu vào rồi chạy tuần tự S1 → S5 |
| Cấu hình | `src/config.py` | Tham số từ `.env` và các trọng số đặt tay |



## 13. Cách chạy bằng Docker

### Yêu cầu

- Docker Desktop;
- nên cấp ít nhất 8 GB RAM cho Docker (Neo4j ~2 GB, pipeline tối đa 5 GB, Airflow ~1,5 GB);
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

Lần đầu, chạy `scripts/verify.sh full --rerun` để dựng kết quả: ETL, nạp graph và huấn
luyện nhiều fold mất khoảng 8–10 phút trên máy 12 nhân với 8 GB RAM cấp cho Docker. Khi lệnh
báo xong, mở:

- Ứng dụng: <http://localhost:7860>
- Airflow: <http://localhost:8080>
- Neo4j Browser: <http://localhost:7474>

Nhấn `Ctrl+C` chỉ dừng theo dõi log, container vẫn chạy.

### Kiểm tra trạng thái

```cmd
docker compose ps
docker compose logs --tail 200 app
docker compose logs --tail 100 neo4j
```

### Chạy pipeline

App chỉ phục vụ kết quả, không tự chạy pipeline. Lần đầu, và mỗi khi đổi dữ liệu hoặc sửa
logic, chạy một trong hai:

```cmd
scripts/verify.sh full --rerun
docker compose exec airflow airflow dags trigger pipeline_khuyen_nghi
```

Cả hai đều dừng app, chạy S1 → S5, bật lại app rồi kiểm tra kết quả. Khi chưa có kết quả
đúng phiên bản, app đứng chờ và log ghi rõ cần chạy lệnh nào.

## 14. Tham số cấu hình

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `DECISION_FREQ` | day | Nhịp ra quyết định: `day`, `week` hoặc `month` |
| `HORIZON` | 20 | Số phiên tương lai dùng để tạo nhãn hành vi |
| `TOPK` | 10 | Số mã mua mới trả về |
| `COOC_WINDOW` | 180 | Cửa sổ ngày tính quan hệ graph |
| `COOC_TOPN` | 25 | Số quan hệ đồng mua mạnh giữ lại mỗi mã |
| `PEER_TOPK` | 20 | Số khách tương tự giữ lại cho mỗi khách |
| `PEER_MIN_SHARED` | 2 | Số mã mua chung tối thiểu để hai khách được coi là tương tự |
| `MARKET_WEIGHT` | 0,60 | Trọng số MarketScore trong điểm mua cuối |
| `BUY_FEE` | 0,0015 | Phí mua tính vào giá vốn khi dựng lại vị thế |
| `BEHAVIOR_HALF_LIFE` | 90 | Chu kỳ bán rã lịch sử giao dịch, đơn vị ngày |
| `SHORT_HOLD_DAYS` | 30 | Bán hết trong số ngày này là bằng chứng ngắn hạn |
| `LONG_HOLD_DAYS` | 90 | Giữ lâu hơn số ngày này là bằng chứng dài hạn |
| `HOLD_EVIDENCE_THRESHOLD` | 0,50 | Tỷ lệ bằng chứng tối thiểu để xếp `SHORT`/`LONG` (tầng 1) |
| `HOLD_MIN_EVIDENCE` | 3 | Số vòng nắm giữ tối thiểu trước khi kết luận |
| `RECENT_WINDOW` | 90 | Cửa sổ ngày đo nhịp giao dịch gần đây |
| `AIRFLOW_ADMIN_USER` | admin | Tài khoản đăng nhập giao diện Airflow |
| `AIRFLOW_ADMIN_PASSWORD` | — | Mật khẩu Airflow, bắt buộc |
| `DUCKDB_MEMORY` | 3GB | Giới hạn RAM DuckDB trong container |
| `GRAPH_WORKERS` | 2 | Số lát cắt thời gian gửi tới Neo4j cùng lúc ở bước đồ thị |
| `DUCKDB_THREADS` | 4 | Số luồng DuckDB; nhiều luồng hơn thì tốn RAM ngoài giới hạn trên |
| `GRADIO_PORT` | 7860 | Cổng giao diện |

Thay đổi feature, `HORIZON`, candidate hoặc trọng số cần chạy lại pipeline để kết quả
đánh giá và model được đồng bộ. Riêng `DECISION_FREQ` nằm trong mã phiên bản của pipeline,
nên đổi nhịp là pipeline tự chạy lại ở lần khởi động kế tiếp.

## 15. Cấu trúc repository

```text
data/                    dữ liệu CSV đầu vào
src/                     source code pipeline và ứng dụng
tests/                   test tự động, chạy trên dữ liệu giả, không cần Neo4j
scripts/                 verify.sh — kiểm tra toàn bộ bằng một lệnh; chốt chặn chỉ số
airflow/dags/            DAG Airflow: kiểm tra code khi có commit, chạy pipeline
AGENTS.md                bản đồ repo, luật cứng và bằng chứng cần nộp khi sửa
docs/                    phương pháp, khảo sát dữ liệu, kết quả chi tiết
artifacts/               DuckDB/model sinh tự động, không commit
docker-compose.yml       Neo4j + app + Airflow
Dockerfile               image chạy Python
requirements.txt         dependency runtime
requirements-dev.txt     công cụ kiểm tra code
pyproject.toml           cấu hình Ruff và pytest
.env.example             mẫu cấu hình, không chứa mật khẩu thật
```


## 16. Kiểm tra chất lượng code

Một lệnh, chạy trong image của project nên không cần cài Python trên máy:

```bash
scripts/verify.sh quick          # lint + test trên dữ liệu giả, vài giây, không cần Neo4j
scripts/verify.sh full           # quick + kiểm tra hệ thống thật đang chạy
scripts/verify.sh full --rerun   # ép chạy lại toàn bộ pipeline (~10 phút) rồi kiểm tra
scripts/verify.sh sample 20      # pipeline thật trên 20% khách, 1–1,5 phút, không cần Neo4j
```

`sample` dùng để thử nhanh một thay đổi: kết quả ghi vào `artifacts/quick/`, đồ thị tính
bằng SQL (`src/s3_graph_sql.py`, cho kết quả khớp từng dòng với Neo4j trên dữ liệu đầy đủ).
Nó chỉ đáng tin cho thay đổi lớn; phép thử chính xác là chạy lại riêng bước S5 trên dữ liệu
đầy đủ với `ABLATE_FEATURES` (xem `AGENTS.md` mục 3).

`full` không chỉ xem app có lên hay không. Nó kiểm tra pipeline đã chạy đúng phiên bản, vị
thế dựng lại khớp snapshot, đồ thị Neo4j khớp đúng bộ dữ liệu và khớp bản SQL, model hơn mức
chọn ngẫu nhiên; rồi gọi hàm gợi ý của app cho 20 khách thật và **đọc lại dữ liệu gốc** để
đối chiếu
(mã MUA có đúng là khuyến nghị PHS còn mở, danh mục có đúng các mã khách đang nắm). Kết
quả ghi vào `artifacts/verify_report.json`; lệnh thoát mã 1 nếu có kiểm tra không qua.

### Airflow: kiểm tra tự động và chạy pipeline

Dự án không dùng CI trên GitHub. `docker compose up -d` khởi động thêm Airflow tại
<http://localhost:8080>, với hai DAG trong `airflow/dags/vv_dags.py`:

| DAG | Khi nào chạy | Các bước |
|---|---|---|
| `ci_kiem_tra_code` | Cứ 5 phút xem có commit mới không; có thì chạy | `lint` → `dinh_dang` → `test` |
| `pipeline_khuyen_nghi` | Bấm chạy tay | `kiem_tra_du_lieu` → `dung_app` → `s1_etl` … `s5_train` → `bat_app` → `kiem_tra_he_thong` → `chot_chan_chi_so` |

```bash
docker compose exec airflow airflow dags trigger pipeline_khuyen_nghi
docker compose exec airflow airflow dags trigger ci_kiem_tra_code
```

Đăng nhập giao diện bằng tài khoản đặt trong `.env`:

```env
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=mat_khau_cua_ban
```

Đổi mật khẩu trong `.env` rồi chạy `docker compose up -d airflow` là mật khẩu được cập nhật.
Thiếu `AIRFLOW_ADMIN_PASSWORD` thì Airflow không khởi động và log báo rõ lý do.

- Mỗi task chạy trong một container riêng tạo từ image của app, nên môi trường giống hệt
  `scripts/verify.sh`. Airflow điều khiển Docker qua một proxy chỉ mở các thao tác cần thiết.
- `pipeline_khuyen_nghi` chạy từng bước S1…S5 thành task riêng: biết ngay bước nào hỏng, mất
  bao lâu, và chạy lại được từ bước hỏng. App bị dừng trong lúc chạy (DuckDB chỉ cho một
  tiến trình ghi) và chỉ bật lại khi mọi bước xong.
- `kiem_tra_he_thong` là đúng các kiểm tra của `verify.sh full`.
- `chot_chan_chi_so` ghi kết quả mỗi lần chạy vào `artifacts/metrics_history.jsonl` và báo đỏ
  nếu HitRate@1 của model hành vi giảm quá 0,02 so với lần trước, hoặc không còn hơn mức
  chọn ngẫu nhiên.
- Phải chạy `docker compose` từ thư mục gốc của dự án: Airflow cần biết đường dẫn dự án
  trên máy chủ (`${PWD}`) để gắn thư mục vào các container task.

Nếu có sẵn môi trường Python:

```bash
python -m pip install -r requirements-dev.txt
ruff check src tests scripts airflow
ruff format --check src tests scripts airflow
python -m pytest
```

`AGENTS.md` ở gốc repo là bản đồ cho người và agent mới vào dự án: thứ gì nằm ở đâu, sửa
loại nào thì phải nộp bằng chứng gì, các luật cứng và các quyết định đã chốt.

Bộ test sinh một bộ dữ liệu giả nhỏ (30 khách, 12 mã, 2 năm) rồi chạy S1 → S2 → S4 → S5
thật trên đó; bước đồ thị được thay bằng SQL tương đương nên **không cần Neo4j** và cả bộ
chạy trong vài giây.

| File test | Kiểm tra gì |
|---|---|
| `tests/test_validate.py` | Thiếu file, thiếu cột, số/ngày sai định dạng, `side` lạ, thiếu VNINDEX đều bị báo |
| `tests/test_positions.py` | Giá vốn bình quân, phí mua, bán hết thì đặt lại; vị thế dựng lại khớp snapshot cuối tháng |
| `tests/test_no_leak.py` | Cắt dữ liệu sau ngày `T` thì đặc trưng tại `t ≤ T` giữ nguyên; nhãn chỉ đến từ tương lai; ứng viên MUA luôn là mã PHS mới BUY đúng ngày |
| `tests/test_metrics.py` | Thước đo xếp hạng, nhóm tầm thường không thổi phồng kết quả, khoảng đệm train/test |
| `tests/test_metric_guard.py` | Chốt chặn chỉ số: giảm nhỏ thì qua, tụt mạnh hoặc không hơn ngẫu nhiên thì đỏ |
| `tests/test_train_smoke.py` | Huấn luyện đầu-cuối, MarketScore không lọt vào model hành vi, app chấm điểm được |

## 17. Giới hạn và lưu ý sử dụng

- Model học khả năng phù hợp/hành vi, không bảo đảm cổ phiếu sẽ sinh lời.
- Chất lượng đầu tư phụ thuộc trực tiếp vào khuyến nghị PHS đầu vào.
- Nếu mã đang nắm không có tín hiệu PHS tại `t`, hệ thống trả `CHƯA CÓ TÍN HIỆU` thay vì
  tự suy diễn BUY/SELL.
- Rổ BUY của PHS rất nhỏ; ở các mốc không có khuyến nghị nào đang mở, nhánh mua mới
  không trả kết quả. Chỉ 1,8% lệnh mua thật của khách rơi vào rổ, nên nhánh MUA trả lời
  câu hỏi "trong vài mã PHS đang khuyến nghị, mã nào hợp khách nhất" chứ không dự đoán
  khách sẽ mua gì.
- Khách hầu như không bán theo lệnh đóng của PHS: xếp tín hiệu PHS lên trước làm HitRate@1
  nhánh danh mục giảm từ 0,503 xuống 0,483. Thứ tự này là quy tắc nghiệp vụ, không phải
  lựa chọn tối ưu theo hành vi.
- Trong danh mục của một khách, model chỉ đoán đúng mã bị bán đầu tiên ở 50% số nhóm so
  với 42% khi chọn ngẫu nhiên. Phần lớn sức dự báo đến từ việc phân biệt khách hay bán với
  khách ít bán, không phải phân biệt các mã của cùng một khách.
- Backtest thực tế cần bổ sung phí giao dịch, thuế, trượt giá, thanh khoản và giới hạn
  tỷ trọng theo mã/ngành.
- Kết quả của hệ thống chỉ phục vụ nghiên cứu và hỗ trợ quyết định, không phải cam kết
  lợi nhuận hay tư vấn đầu tư tự động.

