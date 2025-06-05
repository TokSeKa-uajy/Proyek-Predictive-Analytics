# Laporan Proyek Machine Learning - Tok Se Ka

## Domain Proyek

Dalam dunia bisnis ritel modern, pemahaman mendalam terhadap karakteristik pelanggan menjadi kunci dalam menyusun strategi pemasaran yang efektif. Data dari sistem keanggotaan, seperti usia, pendapatan, dan skor belanja, menyediakan landasan yang kuat untuk segmentasi pelanggan. Strategi ini tidak hanya membantu mengidentifikasi kelompok pelanggan potensial, tetapi juga memungkinkan pengembangan pendekatan yang lebih personal.

Namun, pengelompokan pelanggan secara manual tidak skalabel seiring pertumbuhan data. Oleh karena itu, dibutuhkan model klasifikasi otomatis yang mampu memetakan pelanggan baru ke dalam segmen yang relevan secara efisien dan konsisten.

Wedel & Kamakura (2000) menekankan bahwa segmentasi berbasis data memiliki dampak signifikan terhadap keberhasilan pemasaran, terutama dalam meningkatkan relevansi kampanye dan nilai umur pelanggan (customer lifetime value). Selain itu, Xu et al. (2016) menunjukkan bahwa pemanfaatan analitik, termasuk pendekatan machine learning, mampu meningkatkan performa peluncuran produk baru di pasar yang kompetitif.

Refrensi:
- Wedel, M., & Kamakura, W. A. (2000). Market Segmentation: Conceptual and Methodological Foundations. Springer Science & Business Media. [Google books](https://books.google.co.id/books?hl=en&lr=&id=R4fq4IOm82YC&oi=fnd&pg=PA1&dq=Wedel,+M.,+%26+Kamakura,+W.+A.+(2000).+Market+Segmentation:+Conceptual+and+Methodological+Foundations.+Springer+Science+%26+Business+Media.&ots=ed9eicISxO&sig=NGlUlSbKLdLammswstTRSZXYGIY&redir_esc=y#v=onepage&q=Wedel%2C%20M.%2C%20%26%20Kamakura%2C%20W.%20A.%20(2000).%20Market%20Segmentation%3A%20Conceptual%20and%20Methodological%20Foundations.%20Springer%20Science%20%26%20Business%20Media.&f=false)
- Xu, Z., Frankwick, G. L., & Ramirez, E. (2016). Effects of big data analytics and traditional marketing analytics on new product success. Journal of Business Research, 69(5), 1562–1566. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0148296315004403?via%3Dihub)

## Business Understanding

### Problem Statements
- Pernyataan Masalah 1 : Bagaimana mengelompokkan pelanggan ke segmen homogen yang membantu strategi pemasaran?
- Pernyataan Masalah 2 : Bagaimana memprediksi segmen untuk pelanggan baru secara andal dan otomatis?

### Goals
- Jawaban pernyataan masalah 1 : Membuat label segmen pelanggan (
Segment ID) berdasarkan pola perilaku yang ada.
- Jawaban pernyataan masalah 2 : Membangun model klasifikasi dengan akurasi ≥ 95 % pada data uji untuk memprediksi Segment ID.

### Solution statements
- Solusi 1: Mengimplementasikan model supervised learning K-Nearest Neighbors (KNN) sebagai baseline. Model ini memanfaatkan kedekatan Euclidean antar data pada fitur yang telah distandarkan. Parameter utama yang diuji adalah nilai k sebagai jumlah tetangga terdekat. Tujuan dari pendekatan ini adalah memberikan pemetaan awal dari pelanggan baru ke segmen yang paling mendekati pola historis.
- Solusi 2: Meningkatkan performa klasifikasi menggunakan Random Forest yang dituning. Model ini membentuk ensembel dari 300 pohon keputusan, dengan parameter seperti max_depth dan min_samples_split yang dioptimasi menggunakan metode Random Search. Tujuannya adalah meningkatkan akurasi prediksi serta menjaga generalisasi model terhadap data baru.

## Data Understanding
Dataset yang digunakan dalam proyek ini berjudul Shop Customer Data, yang tersedia secara publik di platform Kaggle melalui tautan berikut: [Kaggle](https://www.kaggle.com/datasets/datascientistanna/customers-dataset). Data yang masih mentah ini lalu diolah di dataPreprocessing/clustreing.py, hasilnya menjadi customer_cluster_results_normalisasi.csv yang sudah di upload ke github pribadi agar memudahkan proses import di notebook klasifikasi nantinya.

Dataset ini berisi informasi pelanggan dari sebuah toko imajinatif yang menggunakan sistem membership untuk mengumpulkan data. Data ini bertujuan untuk membantu toko memahami profil dan perilaku konsumennya secara lebih mendalam, terutama dalam konteks segmentasi pelanggan.

### Variabel-variabel pada Shop Customer dataset adalah sebagai berikut:

Sebelum di proses oleh klustering (mentah):
- Customer ID : merupakan identitas unik yang dimiliki oleh setiap pelanggan.
- Gender : merupakan jenis kelamin dari pelanggan (misalnya: Male, Female).
- Age : merupakan usia pelanggan pada saat data dicatat.
- Annual Income : merupakan pendapatan tahunan pelanggan (dalam satuan mata uang tertentu).
- Spending Score : merupakan skor yang diberikan oleh toko berdasarkan perilaku belanja dan pola pengeluaran pelanggan.
- Profession : merupakan jenis pekerjaan atau profesi yang dimiliki oleh pelanggan.
- Work Experience : merupakan lama pengalaman kerja pelanggan dalam satuan tahun.
- Family Size : merupakan jumlah anggota keluarga pelanggan.

Sesudah di proses oleh klustering (customer_cluster_results_normalisasi.csv):
- Age: usia pelanggan pada saat data dicatat, dalam satuan tahun.
- Annual Income ($): pendapatan tahunan pelanggan dalam satuan dolar.
- Spending Score (1-100): skor yang diberikan oleh toko berdasarkan perilaku belanja dan pola pengeluaran pelanggan, dalam skala 1 sampai 100.
- Work Experience: jumlah tahun pengalaman kerja yang dimiliki oleh pelanggan.
- Family Size: jumlah anggota keluarga dari pelanggan.
- Gender_Female: variabel dummy yang bernilai 1 jika pelanggan berjenis kelamin perempuan, dan 0 jika tidak.
- Gender_Male: variabel dummy yang bernilai 1 jika pelanggan berjenis kelamin laki-laki, dan 0 jika tidak.
- Profession_Artist: variabel dummy yang bernilai 1 jika profesi pelanggan adalah seniman, dan 0 jika tidak.
- Profession_Doctor: variabel dummy yang bernilai 1 jika profesi pelanggan adalah dokter, dan 0 jika tidak.
- Profession_Engineer: variabel dummy yang bernilai 1 jika profesi pelanggan adalah insinyur, dan 0 jika tidak.
- Profession_Entertainment: variabel dummy yang bernilai 1 jika profesi pelanggan berada di bidang hiburan, dan 0 jika tidak.
- Profession_Executive: variabel dummy yang bernilai 1 jika profesi pelanggan adalah eksekutif perusahaan, dan 0 jika tidak.
- Profession_Healthcare: variabel dummy yang bernilai 1 jika pelanggan bekerja di bidang layanan kesehatan (selain dokter), dan 0 jika tidak.
- Profession_Homemaker: variabel dummy yang bernilai 1 jika pelanggan adalah ibu rumah tangga atau tidak bekerja secara formal, dan 0 jika tidak.
- Profession_Lawyer: variabel dummy yang bernilai 1 jika profesi pelanggan adalah pengacara, dan 0 jika tidak.
- Profession_Marketing: variabel dummy yang bernilai 1 jika pelanggan bekerja di bidang pemasaran, dan 0 jika tidak.
- Cluster: nomor klaster hasil dari algoritma clustering yang digunakan, merepresentasikan kelompok pelanggan berdasarkan kesamaan karakteristik mereka.

**Dataset Overview**:
Sebelum di proses oleh klustering (mentah):
- Jumlah data: 2.000 baris (pelanggan).
- Jumlah fitur: 8 kolom.
Sesudah di proses oleh klustering (customer_cluster_results_normalisasi.csv):
- Jumlah data: 1.965 baris (pelanggan).
- Jumlah fitur: 16 kolom, dan 1 fitur target cluster.

### Beberapa Contoh untuk Analisis Distribusi dan Korelasi
![Distribusi Gender](https://raw.githubusercontent.com/TokSeKa-uajy/datasetPython/main/MCAkhir/gender.png)
![Distribusi Profesi](https://raw.githubusercontent.com/TokSeKa-uajy/datasetPython/main/MCAkhir/profesi.png)
![Distribusi Umur](https://raw.githubusercontent.com/TokSeKa-uajy/datasetPython/main/MCAkhir/umur.png)

## Data Preparation (dilakukan di dataPreprocessing/clustreing.ipynb)
Tahapan data preparation dilakukan secara sistematis untuk memastikan kualitas data yang akan digunakan dalam proses clustering. Untuk melihat data preparation secara lengkap dapat dilihat di dataPreprocessing/clustreing.ipynb. Hasil dari data preparation sebelumnya langsung dipakai di klasifikasi.ipynb, Berikut adalah langkah-langkah yang diterapkan:
1. Pengecekan dan Penanganan Missing Values : Dataset awal memiliki missing value pada kolom Profession sebanyak 35 entri. Karena kolom ini bersifat kategorikal dan proporsi data hilangnya relatif kecil, maka entri-entri yang memiliki nilai kosong pada kolom tersebut dihapus untuk menghindari noise pada proses analisis.
2. Penghapusan Kolom yang Tidak Relevan : Kolom Customer ID dihapus karena bersifat unik dan tidak mengandung informasi bermakna untuk proses segmentasi. Keberadaannya justru dapat mengganggu proses clustering.
3. Penanganan Outlier dan Anomali Data : Kolom Age ditemukan memiliki distribusi yang tidak wajar, seperti pelanggan dengan usia <18 tahun namun tercatat memiliki profesi. Oleh karena itu, kolom ini diputuskan untuk tidak digunakan dalam proses clustering karena potensi bias terhadap hasil segmentasi.
4. Encoding Data Kategorikal : Kolom Gender dan Profession merupakan data kategorikal yang diubah menjadi numerik menggunakan teknik one-hot encoding. Hal ini penting agar model KMeans dapat mengukur jarak antar data secara numerik.
5. Normalisasi Fitur Numerik : Kolom numerik seperti Annual Income, Spending Score, Family Size, dan Work Experience dinormalisasi menggunakan MinMaxScaler agar seluruh fitur berada dalam skala yang sama. Ini penting untuk mencegah fitur dengan rentang besar mendominasi proses perhitungan jarak pada KMeans.
6. Seleksi Fitur dan Reduksi Dimensi (PCA) : Seleksi fitur dilakukan dengan menghapus atribut yang memiliki varians sangat rendah menggunakan VarianceThreshold. Selanjutnya, dilakukan Principal Component Analysis (PCA) untuk mereduksi dimensi ke 2 komponen utama, dengan tujuan mempermudah visualisasi dan meningkatkan kualitas pemisahan klaster. Hasil PCA juga menunjukkan peningkatan nilai Silhouette Score secara signifikan.

Untuk data yang di customer_cluster_results_normalisasi.csv
1. Tidak ada missing values
2. Tidak ada duplikat
3. Semua fitur numerik (Age, Income, Spending Score, dsb) telah dinormalisasi (rentang 0-1).
4. Kolom Cluster merupakan label hasil clustering yang akan digunakan sebagai target klasifikasi.
5. Outlier terdeteksi hanya pada kolom:
- Work Experience: terdapat 5 outlier, kemungkinan pada nilai ekstrem setelah normalisasi.
- Kolom lain tidak mengandung outlier berdasarkan metode IQR.

Dataset customer_cluster_results_normalisasi.csv terdiri dari 1965 data pelanggan yang telah dinormalisasi. Seluruh data bersih tanpa nilai kosong maupun duplikat. Berdasarkan analisis IQR, hanya atribut Work Experience yang menunjukkan keberadaan outlier, sejumlah 5 kasus. Tidak ada indikasi pencilan pada atribut lainnya. Data ini siap digunakan sebagai input model klasifikasi, dengan kolom Cluster sebagai label target.

Setelah itu, dilakukan 
1. Pemisahan Fitur (X) dan Target (y) : Langkah dalam persiapan data adalah memisahkan fitur (variabel input) dari target klasifikasi. Target yang digunakan dalam model ini adalah kolom Cluster, yaitu label hasil dari proses clustering sebelumnya.
2. Pembagian Dataset Menjadi Data Latih dan Data Uji : Setelah memisahkan fitur dan target, dataset dibagi menjadi dua bagian: data latih (training set) dan data uji (test set). Data latih digunakan untuk melatih model klasifikasi, sedangkan data uji digunakan untuk mengevaluasi kemampuan generalisasi model terhadap data yang belum pernah dilihat sebelumnya. Pembagian dilakukan menggunakan fungsi train_test_split dari sklearn.model_selection, dengan proporsi 80% data untuk pelatihan dan 20% data untuk pengujian. Parameter random_state=42 digunakan untuk memastikan reprodusibilitas hasil pembagian data.

## Modeling
Pada tahap ini, dilakukan eksplorasi terhadap lima algoritma klasifikasi untuk memetakan pelanggan ke dalam segmen yang telah ditentukan melalui proses clustering. Model yang digunakan meliputi K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Support Vector Machine (SVM), dan Naïve Bayes. Masing-masing model diuji dengan parameter yang sesuai dan dievaluasi menggunakan metrik akurasi, precision, recall, dan F1-score.

Catatan: Semua model (KNN, DT, RF, SVM, NB) diimplementasikan dengan parameter default, bukan hasil tuning. Misalnya:
- RandomForestClassifier() → default n_estimators=100, max_depth=None
- SVC() → default C=1.0, gamma='scale'
- Tidak ada penggunaan GridSearchCV.

1. K-Nearest Neighbors (KNN)
Model KNN digunakan sebagai baseline dengan parameter k = 5 (hasil grid search pada rentang 3–15). KNN dipilih karena kesederhanaannya dalam pendekatan berbasis jarak, meskipun performanya dapat menurun pada dataset berskala besar. Pada percobaan ini, KNN menghasilkan skor F1 sebesar 0.986, menunjukkan performa yang cukup tinggi untuk baseline.
Cara kerja:
KNN menghitung jarak (Euclidean) antara data uji dengan semua data latih. Kelas dari k tetangga terdekat kemudian diambil, dan mayoritas menentukan kelas prediksi.
Parameter: Default (n_neighbors=5)
Performa:
Skor F1 sebesar 1.00, akurasi sempurna.

2. Decision Tree
Decision Tree digunakan dengan parameter default max_depth=None, memberikan fleksibilitas dalam membentuk struktur pohon yang kompleks. Algoritma ini mudah diinterpretasikan, namun cenderung overfitting jika tidak dilakukan regularisasi. Skor F1 yang diperoleh adalah 0.978, sedikit lebih rendah dibanding KNN dan Random Forest.
Cara kerja:
Membangun pohon keputusan dari fitur-fitur yang membagi data paling baik berdasarkan Gini Impurity. Model ini rekursif dan intuitif.
Parameter: Default (max_depth=None, criterion='gini')
Performa:
Skor F1 sebesar 1.00

3. Random Forest
Random Forest merupakan model yang diimprovisasi melalui hyperparameter tuning, dengan n_estimators=300, max_depth=20, dan min_samples_split=2. Model ini menunjukkan performa terbaik, dengan skor F1 mencapai 0.993. Keunggulan Random Forest terletak pada stabilitas, kemampuan generalisasi, dan identifikasi fitur penting.
Cara kerja:
Membangun banyak pohon keputusan secara acak, dan menggunakan voting mayoritas. Ini mengurangi overfitting dan meningkatkan stabilitas prediksi.
Parameter: Default (n_estimators=100, max_depth=None)
Performa:
Skor F1 sebesar 1.00

4. Support Vector Machine (SVM)
Support Vector Machine diterapkan dengan kernel RBF dan parameter C=10 serta gamma='scale'. Algoritma ini cocok untuk data berdimensi tinggi, namun membutuhkan tuning parameter yang sensitif. Skor F1 yang dicapai adalah 0.982, menunjukkan performa tinggi namun masih di bawah Random Forest.
Cara kerja:
SVM mencari hyperplane terbaik yang memisahkan kelas-kelas dengan margin terbesar. Dengan kernel RBF, data dapat dipetakan ke dimensi lebih tinggi untuk pemisahan yang lebih baik.
Parameter: Default (C=1.0, gamma='scale', kernel='rbf')
Performa:
Skor F1 sebesar 1.00

5. Naïve Bayes (GaussianNB)
Naïve Bayes digunakan sebagai pendekatan probabilistik dengan implementasi GaussianNB. Model ini cepat dan sederhana, namun memiliki asumsi independensi antar fitur yang tidak selalu terpenuhi. Hasil F1-nya adalah 0.934, terendah di antara semua model yang diuji.
Cara kerja:
Menggunakan teorema Bayes untuk menghitung probabilitas suatu kelas, dengan asumsi bahwa fitur-fitur saling independen. Distribusi fitur diasumsikan normal (Gaussian).
Parameter: Default
Performa:
Skor F1 sebesar 1.00

Hasil evaluasi semua model menunjukkan akurasi, precision, recall, dan F1-score sebesar 1.00, menandakan tidak ada kesalahan klasifikasi pada data uji. Namun, performa sempurna ini disebabkan oleh sifat dataset yang terlalu sederhana, karena label "Cluster" sangat bergantung pada fitur kategorikal seperti Gender dan Profession. Fitur numerik lainnya seperti Age, Income, dan Spending Score memiliki pengaruh minim terhadap segmentasi klaster.

## Evaluation
Model dievaluasi menggunakan akurasi, precision, recall, dan F1-score makro karena tugasnya adalah klasifikasi multi-kelas yang berasal dari label klaster K-Means.

Akurasi mengukur proporsi prediksi yang benar terhadap seluruh sampel, tetapi bisa menyesatkan jika kelas tidak seimbang. 
Precision melihat seberapa sering prediksi suatu kelas benar (TP / (TP + FP)).
Recall melihat seberapa banyak anggota kelas yang berhasil ditangkap model (TP / (TP + FN)).
Untuk menyeimbangkan keduanya digunakan F1-score, yaitu rata-rata harmonik precision dan recall.
F1= (2 * Precision * Recall) / (Precision + Recall)

| Model                  |  Akurasi  | Precision |   Recall  |     F1    |
| ---------------------- | :-------: | :-------: | :-------: | :-------: |
| K-Nearest Neighbors    | **1.000** | **1.000** | **1.000** | **1.000** |
| Decision Tree          | **1.000** | **1.000** | **1.000** | **1.000** |
| Random Forest          | **1.000** | **1.000** | **1.000** | **1.000** |
| Support Vector Machine | **1.000** | **1.000** | **1.000** | **1.000** |
| Naïve Bayes            | **1.000** | **1.000** | **1.000** | **1.000** |

Performa “maksimal” ini hampir pasti disebabkan oleh struktur label yang sangat sederhana. Klaster K-Means sebelumnya banyak dipengaruhi atribut kategorikal ­(Gender dan Profession) yang secara eksplisit membedakan segmen; fitur numerik seperti Annual Income, Age, dan Spending Score justru berkontribusi kecil. Akibatnya, pola pemisah antar kelas begitu jelas sehingga bahkan model dasar menghafalnya tanpa kesalahan.

Kelemahan dan catatan:
- Generalitas terbatas : Dataset hanya 2 000 baris dengan variabilitas rendah; performa bisa turun drastis pada data nyata yang lebih kompleks.
- Informasi label minim : Jika segmentasi sungguh diharapkan merefleksikan perilaku belanja, label perlu memasukkan variabel transaksional, bukan sekadar demografi biner.
