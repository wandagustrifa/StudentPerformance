# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut adalah institusi pendidikan tinggi yang berkomitmen untuk memberikan pendidikan berkualitas tinggi kepada mahasiswanya. Sebagai institusi edutech, Jaya Jaya Institut menghadapi tantangan dalam mempertahankan tingkat retensi mahasiswa yang optimal.

Dalam industri pendidikan tinggi, tingkat dropout mahasiswa merupakan indikator kritis yang mempengaruhi:
- **Reputasi Institusi**: Tingkat dropout yang tinggi dapat merusak reputasi dan ranking institusi
- **Keberlanjutan Finansial**: Setiap mahasiswa yang dropout berarti kehilangan revenue jangka panjang
- **Efektivitas Program**: Dropout rate yang tinggi mengindikasikan adanya masalah dalam sistem pendidikan
- **Akreditasi**: Lembaga akreditasi mempertimbangkan retention rate dalam penilaian

Dengan data historis menunjukkan tingkat dropout sebesar 32.12%, Jaya Jaya Institut memerlukan sistem prediksi yang dapat mengidentifikasi mahasiswa berisiko tinggi untuk dropout sejak dini.

### Permasalahan Bisnis
Berdasarkan analisis data dan wawancara dengan stakeholder, permasalahan utama yang dihadapi Jaya Jaya Institut adalah:

**1. Tingkat Dropout yang Tinggi**
- 32.12% mahasiswa mengalami dropout, tingkat dropout rate yang sangat tinggi
- Kerugian finansial mencapai miliaran rupiah per tahun
- Dampak reputasi institusi di mata masyarakat dan industri

**2. Kurangnya Sistem Early Warning**
- Tidak ada sistem untuk mengidentifikasi mahasiswa berisiko dropout
- Intervensi akademik dilakukan secara reaktif, bukan proaktif
- Keterlambatan identifikasi menyebabkan intervensi menjadi kurang efektif

**3. Alokasi Sumber Daya yang Tidak Efisien**
- Counselor dan academic advisor tidak memiliki prioritas yang jelas
- Program bantuan akademik tidak tepat sasaran
- Anggaran untuk program retensi tidak dialokasikan secara optimal

**4. Kurangnya Insight Data-Driven**
- Keputusan berbasis intuisi, bukan data
- Tidak ada dashboard untuk monitoring real-time
- Keterbatasan analisis faktor-faktor yang mempengaruhi dropout

### Cakupan Proyek
Proyek ini mencakup pengembangan sistem prediksi dropout mahasiswa yang terdiri dari:

**1. Data Analytics & Machine Learning**
- Pengembangan model prediksi dropout menggunakan Gradient Boosting Classifier
- Feature engineering dan selection untuk 37 variabel predictor
- Model validation dengan accuracy 87.8% dan AUC score 93.07%

**2. Business Intelligence Dashboard**
- Dashboard real-time menggunakan Metabase
- Key Performance Indicators (KPIs) monitoring
- Automated reporting untuk stakeholders

**3. Predictive System Development**
- Web-based application menggunakan Streamlit
- Individual student risk assessment
- Batch prediction untuk cohort analysis
- Risk scoring system untuk prioritization

### Persiapan
**Sumber data**
- **Dataset**: Dicoding Students Performance Dataset
- **Ukuran Dataset**: 4,424 records dengan 37 features
- **Target Variable**: Status (Graduate/Dropout/Enrolled)
- **Data Quality**: Tidak ada missing value dan duplicated value

**Feature Categories**
- **Demographics**: Age, Gender, Marital Status, Nationality
- **Academic**: Admission grades, Previous qualifications, Semester performance
- **Financial**: Scholarship status, Debtor status, Tuition payment
- **Socio-Economic**: Parents' education, Economic indicators

**Setup environment**:
Project ini menggunakan Python 3.8+ dengan library yang tercantum dalam requirements.txt Untuk menginstal semua dependencies, jalankan:

- Setup Environment - Anaconda
```
conda create --name main-ds python=3.9
conda activate main-ds
pip install -r requirements.txt
```

- Setup Environment - Shell/Terminal
```
pip install pipenv
pipenv install
pipenv shell
pip install -r requirements.txt
```

Menjalankan File Prediksi Streamlit
```
streamlit run app.py
```

## Business Dashboard
### Access Information
Link Dashboard: http://localhost:3000/public/dashboard/8fd1c195-994d-4693-ab69-180b8b9856af

URL: http://localhost:3000

Username: root@mail.com

Password: root123

### Dashboard Overview
Business dashboard telah dibuat menggunakan Metabase dengan komponen utama:
- **Overview**
Menampilkan total siswa (4,424), jumlah dropout (1,421), jumlah graduate (2,209) dan jumlah enrolled (794)

- **Distribution Status**
Gambaran umum keberhasilan akademik (49.93% graduate), dropout rate (32.12%), dan enrolled rate (17.95%).

- **Marital Status Impact**
Analisis hubungan antara status pernikahan dan dropout.
Siswa "Legally Separated" memiliki risiko dropout hampir 70%.

- **Gender and Age Analysis**
Distribusi dropout berdasarkan gender (pria: 45%, wanita: 25%) dan kelompok usia (26-30 tahun: 55-60% dropout).

- **Financial Factors**
Pengaruh status keuangan (hutang, beasiswa) terhadap dropout.
Siswa dengan hutang dan keterlambatan pembayaran: 87.4% dropout.

- **Scholarship Analysis**
Siswa penerima beasiswa berpengaruh besar terhadap keberhasilan akademik (Graduated) dan meminimalkan tingkat dropout 

- **Academic Performance**
Perbandingan rata-rata nilai semester 1 dan 2 berdasarkan status (dropout, lulus, aktif).
Prediktor awal: Nilai rendah di semester 1 -> risiko tinggi.

- **Unit Course Per Semester**
Jumlah unit course approve sedikit berisiko tinggi dropout

- **Course Analysis**
- Dropout rate per course: Biofuel course dengan dropoute rate teringgi dengan 66.67% dan Nursing dengan <20% yaitu 15.4%.
- Management kelas malam adalah course dengan jumlah siswa dropout terbanyak
- Identifikasi program dengan kurikulum atau jadwal bermasalah.

- **Early Warning System**
Klasifikasi siswa berdasarkan risiko akademik (Critical, High, Medium, Low) pada semester awal.

### Key Findings
Berdasarkan analisis dashboard, faktor utama yang memengaruhi dropout adalah:

- **Keuangan**
- Siswa dengan hutang dan keterlambatan pembayaran memiliki rate dropout (87.4%).
- Penerima beasiswa memiliki dropout rate sekitar lebih 50% lebih rendah dibanding dengan yang tidak mendapatkan beasiswa.

- **Demografi**
- Pria lebih rentan dropout.
- Siswa 26-30 tahun memiliki risiko 2-3× lebih tinggi daripada siswa <20 tahun.

- **Akademik**
- Nilai semester 1 di bawah 7/15 adalah tanda bahaya.
- Course dengan jadwal malam (Management kelas malam) dan course seperti Biofuel bermasalah.

- **Dukungan Non-Akademik**
Siswa dengan status legally-separated memiliki dropout rate tertinggi (hampir 70%).

## Menjalankan Sistem Machine Learning
### Prototype System Access
Web Application (Streamlit)

URL: https://studentperformance-prediction.streamlit.app/

Platform: Streamlit-based interactive application

**1. Cara Menjalankan app**:
```
streamlit run app.py
```

**2. Access Dashboard**:
Open browser ke http://localhost:8501
Navigate melalui sidebar menu:

🏠 Dashboard: Overview metrics

🔮 Prediction: Individual/batch prediction

📊 Analytics: Feature importance analysis

ℹ️ About: Documentation

### Features Aplikasi
**1. Dashboard Page**
Overview metrics dan visualisasi
Real-time student statistics
Performance trends analysis

**2. Prediction Page**
Individual Prediction: Input manual data mahasiswa
Batch Prediction: Upload CSV untuk multiple predictions
Risk Scoring: Analisis risiko untuk cohort tertentu

**3. Analytics Page**
Feature Importance: Faktor-faktor yang paling berpengaruh
Correlation Analysis: Hubungan antar variabel
Demographics Insights: Analysis berdasarkan demografis

**4. About Page**
Dokumentasi model dan dataset
Technical specifications
Usage guidelines

### Model Performance Metrics:
Accuracy: 87.8%
AUC Score: 93.07%
Precision (Dropout): 86%
Recall (Dropout): 74%
F1-Score: 80%

### Key Features (Top 10):
1. Curricular units 2nd semester approved
2. Curricular units 1st semester approved
3. Tuition fees up to date
4. Age at enrollment
5. Curricular units 1st semester grade
6. Admission grade
7. Previous qualification grade
8. Curricular units 2nd semester grade
9. Scholarship holder
10. Debtor status

## Conclusion
Proyek Student Dropout Prediction System telah berhasil mengembangkan solusi komprehensif untuk mengatasi permasalahan tingkat dropout yang tinggi di Jaya Jaya Institut.

### Key Achievements:
**1. Predictive Model Excellence**
Achieved 87.8% accuracy dalam prediksi dropout risk
AUC score 93.07% menunjukkan excellent discrimination capability
Model dapat mengidentifikasi 74% mahasiswa berisiko dropout

**2. Business Intelligence Implementation**
**Masalah Utama**:
- Keuangan adalah penyebab utama dropout (hutang, kurang beasiswa).
- Kurikulum tidak relevan seperti biofuel dan jadwal tidak fleksibel memperburuk retensi.
- Siswa dengan tantangan personal (usia dewasa, status pernikahan kompleks) butuh pendekatan khusus.

**Peluang Perbaikan**:
- Program dengan prospek karir jelas (Nursing) memiliki retensi tinggi (<20% dropout).
- Intervensi akademik di semester awal efektif mencegah dropout.

### Rekomendasi Action Items
**Prioritas 1**: Penanganan Keuangan
- Luncurkan program beasiswa darurat untuk 500 siswa berisiko tahun depan.

**Prioritas 2**: Perbaikan Akademik
- Remedial intensif untuk siswa dengan nilai semester 1 kurang dari 7/15.
- Redesign kurikulum untuk program dengan jumlah dan tingkat dropout tinggi (Biofuel, Management Malam).

**Prioritas 3**: Dukungan Siswa
- Konseling 24/7 untuk masalah keluarga/keuangan.
- Kelas yang jadwalnya fleksibel.

**Prioritas 4**: Kolaborasi Industri
- Program magang berbayar untuk meningkatkan relevansi pembelajaran.


