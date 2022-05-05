# Bag of Tricks and A Strong ReID Baseline

- Nhóm sử dụng source code của tác giả, tiến hành chạy thử nghiệm, train lại và đánh giá, so sánh giữa các phương pháp với nhau
- Thông tin chi tiết [source code gốc](https://github.com/michuanhaohao/reid-strong-baseline)

## Quá trình train

1. `cd` di chuyển đến thư mục cần chạy

2. Chạy lệnh `git clone https://github.com/viettham1998/reid-strong-baseline.git`

3. Cài đặt các thư viện:
    - [pytorch>=0.4](https://pytorch.org/)
    - torchvision
    - [ignite=0.1.2](https://github.com/pytorch/ignite)
    - [yacs](https://github.com/rbgirshick/yacs)

4. Chuẩn bị dữ liệu

    Nhóm tiến hành sử dụng 02 bộ dataset được Download từ 02 nguồn sau:

    （1）[Market1501](http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)

    * Giải nén và đổi tên thư mục giải nén thành `market1501`. Cấu trúc của thư mục sẽ là:

    ```bash
    data
        market1501 # Trong thư mục có chứa 6 files.
            bounding_box_test/
            bounding_box_train/
            ......
    ```
    （2）[DukeMTMC-reID](https://github.com/layumi/DukeMTMC-reID_evaluation#download-dataset)

    * Giải nén và đổi tên thư mục giải nén thành `dukemtmc-reid`. Cấu trúc của thư mục sẽ là:

    ```bash
    data
        dukemtmc-reid
        	DukeMTMC-reID # Trong thư mục có chứa 8 files.
            	bounding_box_test/
            	bounding_box_train/
            	......
    ```

5. Thay thế đường dẫn đến thư mục Pretrain tương ứng

    Tại các file: 
    - configs/baseline.yml
    - configs/softmax.yml
    - configs/softmax_triplet.yml
    - configs/softmax_triplet_with_center.yml
	
    Thay thế Pretrain_Path trỏ đến đúng file tương ứng, ở đây, trong đồ án này, nhóm sử dụng file `resnet50-19c8e357.pth`
    
	Thay thế OUTPUT_DIR thành thư mục chứa dữ liệu sau khi train

## Quá trình Train
Chạy trong terminal

1. Market1501, cross entropy loss + triplet loss

```bash
python tools/train.py --config_file= D:/hvtham/jupyter/reid-strong-baseline/configs/softmax_triplet_with_center.yml MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('data')" OUTPUT_DIR "('demo/outputMarketWithCenter')"
```

2. DukeMTMC-reID, cross entropy loss + triplet loss + center loss


```bash
python tools/train.py --config_file=D:/hvtham/jupyter/reid-strong-baseline/configs/softmax_triplet_with_center.yml MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('dukemtmc')" DATASETS.ROOT_DIR "('data')" OUTPUT_DIR "('demo/outputDukeWithCenter')"
```

## Quá trình Test
Test thử trên file train ở bước trên

```bash
python tools/test.py --config_file=D:/hvtham/jupyter/reid-strong-baseline/configs/softmax_triplet_with_center.yml MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('data')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('./outputMarketWithoutCenter/resnet50_model_80.pth')"
```
Đối với việc so sánh giữa kết hợp cross entropy loss + triplet loss hay chỉ mình cross entropy loss thì các bước chạy vẫn như trên, chỉ thay đổi đường dẫn --config_file đến file yml tương ứng.

Trên github này cũng có chứa những [file log](https://github.com/viettham1998/reid-strong-baseline/tree/main/logs) mà nhóm đã chạy.
