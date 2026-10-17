"""Bài 1: mở rộng Bayes cho ba nguyên nhân loại trừ nhau."""


def main():
    cac_nhom = ["Hoc deu", "Hoc don", "It hoc"]
    tien_nghiem = [0.50, 0.30, 0.20]
    ty_le_qua = [0.95, 0.70, 0.30]
    # Xác suất toàn phần: cộng xác suất đồng thời của cả ba nhánh.
    dong_thoi = [prior * qua for prior, qua in zip(tien_nghiem, ty_le_qua)]
    p_qua = sum(dong_thoi)
    hau_nghiem = [p / p_qua for p in dong_thoi]
    print("BAI 1 - BAYES VOI BA NGUYEN NHAN")
    print(f"P(qua mon) = {p_qua:.4f}")
    for ten, p in zip(cac_nhom, hau_nghiem):
        print(f"P({ten} | qua) = {p:.4f}")
    print(f"Tong ba xac suat = {sum(hau_nghiem):.4f}")
    print("Nhan xet: biet da qua mon lam xac suat thuoc nhom hoc deu tang len.")


if __name__ == "__main__":
    main()
