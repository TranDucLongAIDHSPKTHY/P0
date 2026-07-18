# BÁO CÁO THỐNG KÊ VÀ KIỂM CHỨNG TAXONOMY

**Dự án:** ID-GRec verify TaxPro-CL  
**Ngày tổng hợp:** 15/07/2026  
**Phạm vi:** Amazon-Book và Yelp2018  
**Trạng thái tổng thể:** **PASS (Gate G1)**

## 1. Tóm tắt điều hành

Pipeline đã hợp nhất metadata nhiều phiên bản, ánh xạ metadata vào `remap_id` của ID-GRec, tạo taxonomy trên train sau lọc 5-core, chọn category chính theo chiến lược `primary`, gộp leaf nhỏ khi có ancestor phù hợp, và xuất mapping item–category dùng cho mô hình.

| Chỉ tiêu                           |        Amazon-Book |          Yelp2018 |
| ---------------------------------- | -----------------: | ----------------: |
| Item trong mapping nguồn           |             91.599 |            38.048 |
| Metadata khớp trên toàn mapping    | 91.599 (100,0000%) | 36.729 (96,5333%) |
| Item active sau 5-core             |             88.416 |            37.381 |
| Item active có taxonomy hợp lệ     | 88.416 (100,0000%) | 36.283 (97,0627%) |
| Bản ghi trong `item2category` cuối |             88.416 |            37.381 |
| Category node/path sau xử lý       |                635 |               739 |
| Leaf category sau xử lý            |                626 |               739 |
| Độ sâu taxonomy trung bình cuối    |           3,326592 |          1,000000 |
| Số path trung bình/item cuối       |           1,000000 |          1,000000 |
| Gate G1                            |           **PASS** |          **PASS** |

Kết luận chính:

- Mapping ID nguồn → `remap_id` của cả hai miền là duy nhất, liên tục từ 0 và không có dòng lỗi/trùng.
- Mapping cuối bao phủ đúng 100% tập item active về mặt khóa: không thiếu khóa, không thừa khóa, không có key khác `item_id`.
- Amazon đạt 100% coverage taxonomy trên tập active.
- Yelp có 1.098 item active chưa có category; semantic coverage là 97,0627%, dù file mapping vẫn có đủ 37.381 khóa.
- Amazon giảm 1.585 leaf xuống 626 leaf bằng cách đưa 2.785 item từ leaf nhỏ lên ancestor. Yelp giữ nguyên 739 leaf do taxonomy nguồn phẳng, depth 1.
- Taxonomy và tần suất category chỉ được xây từ train; test không tham gia xây taxonomy. Không có overlap train–validation, train–test hoặc validation–test ở cả ba seed.

## 2. Nguồn dữ liệu và sản phẩm được kiểm tra

### Mapping ID nguồn

- Amazon: `dataset/amazon-book/item_list.txt`, ánh xạ ASIN → `remap_id`.
- Yelp: `dataset/yelp2018/item_list.txt`, ánh xạ business ID → `remap_id`.

### Metadata và quyết định merge

- `metadata/merged_metadata_{amazon,yelp}.json`
- `metadata/merge_decisions_{amazon,yelp}.json`
- `metadata/merge_summary_{amazon,yelp}.json`

### Mapping và kết quả cuối

- `metadata/item2category_{amazon,yelp}.json`
- `preprocessed/preprocessing_summary.{json,md}`
- `preprocessed/taxonomy_distribution_{amazon,yelp}.png`
- `dataset_verify/{amazon-book,yelp2018}/{train,validation,test}.txt`

Mỗi bản ghi `item2category` có các trường `item_id`, `leaf_category`, `taxonomy_path`, `leaf_categories`, `taxonomy_paths`, `original_leaf_category`, `original_taxonomy_path` và `category_strategy`. Các trường `original_*` lưu taxonomy trước gộp leaf; các trường còn lại là kết quả cuối dùng cho mô hình.

## 3. Phương pháp xây taxonomy

### 3.1. Hợp nhất metadata

Pipeline đọc mapping `org_id → remap_id` và kiểm tra tính duy nhất của cả hai phía. Khi một item có nhiều candidate metadata, thứ tự ưu tiên là:

1. Taxonomy sâu hơn (`deeper_taxonomy`).
2. Nếu cùng độ sâu, nhiều taxonomy path hơn (`more_taxonomy_paths`).
3. Nếu vẫn hòa, metadata mới hơn (`newer_metadata`).
4. Nếu chỉ có một candidate, dùng candidate đó (`only_candidate`).
5. Nếu không có candidate, đánh dấu `missing` và để taxonomy rỗng.

Amazon giữ cấu trúc phân cấp. Yelp tách `CATEGORIES` thành các category độc lập, mỗi category là một path dài 1. Path được làm sạch, bỏ rỗng và khử trùng lặp không phân biệt hoa/thường.

### 3.2. Xử lý cho mô hình

- Lọc 5-core lặp đến hội tụ chỉ trên train.
- Chỉ item còn active được đưa vào mapping cuối.
- Strategy `primary` chọn một path/item: path sâu nhất; nếu hòa, path xuất hiện sớm hơn.
- Leaf có ít hơn 10 item được đưa lên ancestor cho tới khi đạt ngưỡng hoặc chỉ còn depth 1.

Metadata ban đầu trung bình có 1,901909 path/item ở Amazon và 4,173569 path/item ở Yelp, nhưng mapping cuối chỉ giữ một primary path/item. Vì vậy mapping cuối là single-category, không biểu diễn đầy đủ multi-label nguồn.

## 4. Thống kê metadata merge

### 4.1. Amazon-Book

| Chỉ tiêu                            |    Giá trị |
| ----------------------------------- | ---------: |
| Tổng item mapping                   |     91.599 |
| Có metadata / thiếu metadata        | 91.599 / 0 |
| Coverage                            |  100,0000% |
| Độ sâu trung bình tại bước merge    |   3,382602 |
| Path trung bình/item tại bước merge |   1,901909 |

| Phiên bản được chọn | Số item |    Tỷ lệ |
| ------------------- | ------: | -------: |
| 2014                |  21.030 | 22,9588% |
| 2018                |   3.843 |  4,1955% |
| 2023                |  66.726 | 72,8458% |

| Lý do chọn            | Số item |
| --------------------- | ------: |
| `newer_metadata`      |  51.224 |
| `deeper_taxonomy`     |  20.616 |
| `only_candidate`      |  19.464 |
| `more_taxonomy_paths` |     295 |

Có 19.464 item với 1 candidate, 19.105 item với 2 candidate và 53.030 item với 3 candidate.

### 4.2. Yelp2018

| Chỉ tiêu                         |        Giá trị |
| -------------------------------- | -------------: |
| Tổng item mapping                |         38.048 |
| Có metadata / thiếu metadata     | 36.729 / 1.319 |
| Coverage toàn mapping            |       96,5333% |
| Độ sâu trung bình tại bước merge |       0,999755 |
| Path trung bình/item có metadata |       4,173569 |

| Phiên bản được chọn | Số item |
| ------------------- | ------: |
| 2018                |  36.725 |
| 2021                |       2 |
| 2022                |       2 |

| Lý do chọn            | Số item |
| --------------------- | ------: |
| `only_candidate`      |  36.725 |
| `missing`             |   1.319 |
| `more_taxonomy_paths` |       2 |
| `newer_metadata`      |       2 |

Có 36.725 item với 1 candidate, 4 item với 2 candidate và 1.319 item không có candidate. Metadata Yelp 2021/2022 chỉ bổ sung được 4 trường hợp.

## 5. Ảnh hưởng của lọc 5-core

| Miền   | User trước → sau | Item trước → sau | Interaction trước → sau | Vòng hội tụ |
| ------ | ---------------: | ---------------: | ----------------------: | ----------: |
| Amazon |  52.643 → 52.642 |  91.599 → 88.416 |   2.380.730 → 2.372.615 |           3 |
| Yelp   |  31.668 → 31.668 |  38.048 → 37.381 |   1.237.259 → 1.235.307 |           2 |

- Amazon loại 3.183 item và 8.115 interaction; cả 88.416 item còn lại có taxonomy.
- Yelp loại 667 item và 1.952 interaction. Trong 37.381 item còn lại, 36.283 có taxonomy và 1.098 không có taxonomy.
- Trong 1.319 item Yelp thiếu metadata toàn cục, 221 item không còn active sau 5-core; 1.098 item thiếu vẫn active.

## 6. Taxonomy trước và sau gộp leaf

### 6.1. Amazon-Book

| Chỉ tiêu                   | Trước gộp |    Sau gộp |
| -------------------------- | --------: | ---------: |
| Category node/path         |     1.931 |        635 |
| Leaf                       |     1.585 |        626 |
| Item có taxonomy           |    88.416 |     88.416 |
| Kích thước leaf trung bình | 55,782965 | 141,239617 |
| Độ sâu trung bình          |  3,365986 |   3,326592 |
| Path trung bình/item       |  1,000000 |   1,000000 |

- Leaf giảm 959 (60,50%); 2.785 item, khoảng 3,15% item active, được chuyển lên ancestor.
- Leaf lớn nhất sau gộp: `Books > Literature & Fiction > Genre Fiction`, 10.360 item (khoảng 11,72% item active).
- Leaf nhỏ nhất: `Health & Personal Care`, 1 item. Leaf này ở depth 1 nên không còn ancestor để gộp.

### 6.2. Yelp2018

| Chỉ tiêu                   | Trước gộp |   Sau gộp |
| -------------------------- | --------: | --------: |
| Category node/path         |       739 |       739 |
| Leaf                       |       739 |       739 |
| Item có taxonomy           |    36.283 |    36.283 |
| Kích thước leaf trung bình | 49,097429 | 49,097429 |
| Độ sâu trung bình          |  1,000000 |  1,000000 |
| Path trung bình/item       |  1,000000 |  1,000000 |

- Không item nào được gộp lên ancestor.
- Leaf lớn nhất: `Restaurants`, 6.641 item (khoảng 18,30% item có taxonomy).
- Leaf nhỏ nhất: `General Festivals`, 1 item.
- `min_leaf_size=10` không thể xử lý leaf nhỏ vì toàn bộ path Yelp ở depth 1. Đây là giới hạn cấu trúc, không phải lỗi thực thi.

`total_categories` là số node được xác định bởi các prefix path duy nhất, không đơn thuần là số chuỗi nhãn category khác nhau.

## 7. Top leaf category cuối

### Amazon-Book

| Hạng | Taxonomy path                                               |   Item |
| ---: | ----------------------------------------------------------- | -----: |
|    1 | Books > Literature & Fiction > Genre Fiction                | 10.360 |
|    2 | Books > Mystery, Thriller & Suspense > Thrillers & Suspense |  4.255 |
|    3 | Books > Mystery, Thriller & Suspense > Mystery              |  3.913 |
|    4 | Books > Literature & Fiction > United States                |  3.773 |
|    5 | Books > Christian Books & Bibles > Literature & Fiction     |  2.760 |
|    6 | Books > Science Fiction & Fantasy > Fantasy                 |  2.628 |
|    7 | Books > Teen & Young Adult > Literature & Fiction           |  2.354 |
|    8 | Books > Romance > Contemporary                              |  2.312 |
|    9 | Books                                                       |  2.051 |
|   10 | Books > Literature & Fiction > Action & Adventure           |  1.662 |

### Yelp2018

| Hạng | Category               |  Item |
| ---: | ---------------------- | ----: |
|    1 | Restaurants            | 6.641 |
|    2 | Food                   | 2.219 |
|    3 | Nightlife              | 1.045 |
|    4 | Bars                   |   919 |
|    5 | Shopping               |   774 |
|    6 | Mexican                |   727 |
|    7 | American (Traditional) |   698 |
|    8 | Coffee & Tea           |   623 |
|    9 | Chinese                |   587 |
|   10 | Pizza                  |   568 |

Phân phối category lệch đáng kể; nên cân nhắc category-balanced sampling hoặc weighting nếu taxonomy tham gia trực tiếp vào loss.

## 8. Mapping item–category verified

Kiểm toán read-only đã đối chiếu trực tiếp `item2category_*.json` với hợp item trong `dataset_verify/{train,validation}.txt`.

| Kiểm tra                       | Amazon |   Yelp |
| ------------------------------ | -----: | -----: |
| Khóa mapping cuối              | 88.416 | 37.381 |
| Item active đối chiếu          | 88.416 | 37.381 |
| Khóa active thiếu / khóa thừa  |  0 / 0 |  0 / 0 |
| Key khác `item_id`             |      0 |      0 |
| Item có path hợp lệ            | 88.416 | 36.283 |
| Item có path rỗng              |      0 |  1.098 |
| `leaf_category` lệch cuối path |      0 |      0 |
| Bản ghi không dùng `primary`   |      0 |      0 |
| Item đổi path do gộp leaf      |  2.785 |      0 |

| Kiểm tra mapping nguồn |   Amazon |     Yelp |
| ---------------------- | -------: | -------: |
| Dòng hợp lệ            |   91.599 |   38.048 |
| Dòng lỗi hoặc ID trùng |        0 |        0 |
| Dải `remap_id`         | 0–91.598 | 0–38.047 |
| Dải liên tục           |       Có |       Có |

Mapping cuối **đúng về khóa và nhất quán nội bộ**. Với Yelp, 1.098 path rỗng là thiếu metadata/category đã biết, không phải lỗi lệch ID.

## 9. Kiểm chứng leakage và tính toàn vẹn split

- Metadata chỉ đọc từ `merged_metadata_{domain}.json`.
- 5-core, tần suất taxonomy và gộp leaf chỉ dùng train.
- Test chỉ dùng kiểm tra overlap/hash, không dùng chọn category.
- Mỗi split giữ bản sao nguyên byte của test gốc.

| Miền   | Seed |     Train | Validation | Tỷ lệ validation | Train∩Val | Train∩Test | Val∩Test |
| ------ | ---: | --------: | ---------: | ---------------: | --------: | ---------: | -------: |
| Amazon |   42 | 2.130.879 |    241.736 |       10,188589% |         0 |          0 |        0 |
| Amazon |  123 | 2.130.879 |    241.736 |       10,188589% |         0 |          0 |        0 |
| Amazon | 2026 | 2.130.879 |    241.736 |       10,188589% |         0 |          0 |        0 |
| Yelp   |   42 | 1.108.976 |    126.331 |       10,226689% |         0 |          0 |        0 |
| Yelp   |  123 | 1.108.976 |    126.331 |       10,226689% |         0 |          0 |        0 |
| Yelp   | 2026 | 1.108.976 |    126.331 |       10,226689% |         0 |          0 |        0 |

SHA-256 test được bảo toàn:

- Amazon: `6250754994dd953b91b48dccdd77a73f318e0634b4757c2abfcc9d33a54ab0f5`
- Yelp: `9cd9b37926cd1447a06625ffa7914527573fb097c10cb0bbdb1d1b8ded3742fe`

Hash trước/sau, `dataset_verify/test.txt` và test trong ba seed đều khớp nguồn. Với ngưỡng coverage G1 là 80%, Amazon đạt 100% và Yelp đạt 97,0627%; cả hai đều **PASS**.

## 10. Hạn chế và khuyến nghị

1. **Phân biệt key coverage và semantic coverage:** Yelp có đủ 37.381 khóa nhưng chỉ 36.283 khóa có category; không nên báo cáo taxonomy coverage là 100%.
2. **Yelp bị ép từ multi-label về single-label:** trung bình nguồn có 4,173569 category/item, nhưng `primary` giữ category xuất hiện sớm nhất do tất cả path cùng depth 1.
3. **Ngưỡng leaf 10 không tuyệt đối:** thuật toán dừng ở depth 1, nên vẫn có leaf dưới 10 item ở cả hai miền, đặc biệt Yelp.
4. **Amazon phù hợp hơn cho mô hình phân cấp:** depth trung bình cuối là 3,326592; Yelp không có quan hệ ancestor–descendant.
5. **Mất cân bằng category:** leaf lớn nhất chiếm khoảng 11,72% item active Amazon và 18,30% item có taxonomy Yelp.
6. **Item Yelp thiếu taxonomy:** nên giữ nhãn `unknown`, loại khỏi taxonomy-aware loss nhưng vẫn giữ cho CF, hoặc bổ sung metadata; không nên gán category phổ biến nhất một cách ngầm định.
7. **Mapping phụ thuộc cấu hình train:** khi thay split, core size, strategy hoặc `min_leaf_size`, cần tái sinh và kiểm chứng mapping.

## 11. Kết luận

- **Amazon-Book:** coverage đầy đủ, taxonomy phân cấp rõ, mapping item–category nhất quán. Gộp leaf giảm mạnh độ phân mảnh mà không mất coverage.
- **Yelp2018:** mapping ID chính xác và coverage vượt ngưỡng, nhưng taxonomy cuối là single-label phẳng; 1.098 item active chưa có category và cơ chế gộp hiện tại không xử lý được leaf depth 1 nhỏ.
- **Tính hợp lệ thí nghiệm:** cả hai miền đạt Gate G1, không có leakage và test được bảo toàn nguyên byte.

Dữ liệu hiện tại đủ điều kiện cho thí nghiệm taxonomy-aware. Khi báo cáo kết quả mô hình, cần công bố riêng semantic coverage Yelp 97,0627% và giới hạn single-label/phẳng.

## 12. Căn cứ tái lập

Số liệu được tổng hợp từ `preprocessing_logs/prepare_metadata.log`, `preprocessing_logs/preprocessing.log`, các file `merge_summary`, `merge_decisions`, `item2category`, `preprocessed/preprocessing_summary.json`, hai file `item_list.txt` và các split trong `dataset_verify`.
