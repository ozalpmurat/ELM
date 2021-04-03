# Extreme Learning Machine (ELM) Uygulaması
Tek gizli katmandan oluşan temiz bir Extreme Learning Machine (ELM) yapısı var. Hücre sayısı kod içerisinde belirlenebiliyor. sin(x) verilerini tahmin etmeye çalışan bir ELM yapısı kurgulandı. Dolayısıyla; sınıflandırma değil regresyon yapıyor.

ELM yapısında iterasyon veya geri yayılım gibi işlemler bulunmamaktadır. Giriş ağırlıkları rastgele belirlendikten sonra, gizli katman çıkışındaki β ağırlıkları numerik olarak hesaplanmaktadır. Burada ters matris işlemi için Pseudo-Inverse yöntemi kullanılır. Modelin fit edilmesi (eğitilmesi) sırasında hesaplanan tek parametre, β vektörüdür. Onun dışındaki parametreler, ilk atandığı zamanki gibi kullanılır. İterasyon olmadığı için çok hızlı eğitilen bir YSA türüdür.

Kodların orjinal kaynağı:
https://github.com/alanrabelo/Extreme-Learning-Machines/

# Konsol çıktısı
Sin(x) tahmini için, 1.57 (0.5 pi) olarak verilen bir giriş için hesaplanan çıktı 0.42 olmuştur
Sin(1.57) için gerçek değer ise 1.00 olmalıdır
Eğitilen sistem, %58.08 hata yapmıştır.

# Grafik çıktısı
