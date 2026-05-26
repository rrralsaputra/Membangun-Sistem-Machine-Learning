# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    input_path = "Eksperimen_SML_Muhammad_Geralldo/heart_raw.csv"
    output_dir = "Eksperimen_SML_Muhammad_Geralldo/preprocessing/heart_disease_preprocessing"
    output_path = os.path.join(output_dir, "heart_preprocessing.csv")

    # Membaca dataset
    df = pd.read_csv(input_path)

    print("Dataset berhasil dimuat")
    print(f"Ukuran dataset awal: {df.shape}")

    # Menghapus data duplikat
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Jumlah data duplikat yang dihapus: {duplicate_count}")

    # Menghapus missing value
    missing_count = df.isnull().sum().sum()
    df = df.dropna()
    print(f"Jumlah missing value yang ditangani: {missing_count}")

    # Menentukan kolom target
    target_column = "target"

    if target_column not in df.columns:
        raise ValueError(f"Kolom target '{target_column}' tidak ditemukan.")

    # Memisahkan fitur dan target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Standarisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Membuat dataframe hasil preprocessing
    df_preprocessing = pd.DataFrame(X_scaled, columns=X.columns)
    df_preprocessing[target_column] = y.values

    # Membuat folder output jika belum ada
    os.makedirs(output_dir, exist_ok=True)

    # Menyimpan dataset hasil preprocessing
    df_preprocessing.to_csv(output_path, index=False)

    print("Preprocessing selesai")
    print(f"Ukuran dataset akhir: {df_preprocessing.shape}")
    print(f"Dataset hasil preprocessing disimpan di: {output_path}")


if __name__ == "__main__":
    main()