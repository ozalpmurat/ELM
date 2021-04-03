%% AÇIKLAMA
% Tek gizli katmandan oluşan bir temiz Extreme Learning Machine (ELM)
% yapısı var. Hücre sayısı kod içerisinde belirlenebiliyor. sin(x)
% verilerini tahmin etmeye çalışan bir ELM yapısı kurgulandı.
% 
% ELM yapısında iterasyon veya geri yayılım gibi işlemler bulunmamaktadır.
% Giriş ağırlıkları rastgele belirlendikten sonra, gizli katman çıkışındaki
% β ağırlıkları numerik olarak hesaplanmaktadır. Burada ters matris işlemi
% için Pseudo-Inverse yöntemi kullanılır. Modelin fit edilmesi (eğitilmesi)
% sırasında hesaplanan tek parametre, β vektörüdür. Onun dışındaki
% parametreler, ilk atandığı zamanki gibi kullanılır. İterasyon olmadığı
% için çok hızlı eğitilen bir YSA türüdür.
% 
% Kodların orjinal kaynağı:
% https://github.com/alanrabelo/Extreme-Learning-Machines/blob/master/main.m
%
% Murat Özalp

%% Temizle
clear variables;
close all;

%% 1 Giriş (X) ve 1 çıkış (Y) için veri seti (sin_x) oluştur
X = linspace(-2*pi,2*pi,1500);
Y = sin(X);
subplot(3,1,1); plot(X,Y);
title('-2\pi ve 2\pi aralığında Sin(X) grafiği')
set(gca,'XTick',-2*pi:pi/2:2*pi) 
set(gca,'XTickLabel',{'-2*\pi','-3*\pi/2','-\pi','-\pi/2','0','\pi/2','\pi','3*\pi/2','2*\pi'})

% Y Verilerine gürültü ekle. Tam gerçek sin(x) verileri yerine yaklaşık
% verilerle eğitim  test yapacağız.
gurultu = rand(1, length(Y)) .* 0.3 - 0.1; %-0.1 değerini ben ekledim. Bunu koymayınca, hep sin(x) eğrisinin üstünde oluyor gürültüler.
Y_Orjinal=Y;
Y = Y + gurultu;
subplot(3,1,2); scatter(X,Y,'.')
hold on; plot(X,Y_Orjinal,'r');
title('Sin(X) verilerine gürültü ekledik')

%% Eğitim ve test verilerini ayır
% X sayısı kadar rastgele sayı oluşturup randIndexes vektörüne atıyoruz.
% Daha sonra; randIndexes'in ilk %80 değerine göre veri setinden indexleyerek değer çekiyoruz.
randIndexes = randperm(length(X)); % X sayısı kadar rasgele sayı üret
X_egitim = X(randIndexes(1:floor(length(X) * 0.8))); % Rasgele %80 eğitim verisi
X_test = X(randIndexes(floor(length(X) * 0.8) + 1:end)); % Gerisi test verisi
Y_egitim = Y(randIndexes(1:floor(length(Y) * 0.8)));
Y_test = Y(randIndexes(floor(length(Y) * 0.8) + 1:end));

%% Eğitim hesap
HucreSayisi=3;
W = rand(HucreSayisi,1); % Giriş katmanı ağırlıklarını rastgele belirle
NetGirisler = W * X_egitim; % Ağırlıklar ile girişleri çarp. Aktivasyon fonksiyonu için net girişleri hesapla
H = 1 ./ (1 + exp(-NetGirisler)); % Aktivasyon fonksiyonu olarak Sigmoid kullandık. H (Huang) matrisini elde ettik.
inverse = pinv(H); % Ters matris işlemi (Pseudo-Inverse, Moore–Penrose Inverse)
Beta = Y_egitim * inverse; % Beta değerleri bulundu. Model fit edilmiş (eğitilmiş) oldu.

%% Test hesap
NetGirisTest = W * X_test; % Ağırlıklar ile girişleri çarp. Aktivasyon fonksiyonu için net girişleri hesapla
Htest = 1 ./ (1 + exp(-NetGirisTest)); % Aktivasyon fonksiyonu olarak Sigmoid kullandık. H (Huang) matrisini elde ettik.
Y_tahmin = Beta * Htest; % H matrisi ile Beta ağırlıkları çarpılarak test çıkışları elde edildi.
MSE = sqrt(sum((Y_tahmin - Y_test).^2)); % Hatasız YSA olmaz. Ortalama kare hata hesabı.

%% Çıkarım
% sin(x) fonksiyonu için çıkarım yapmak istiyoruz. Bunun için OrnekX
% şeklinde bir değişkeni giriş olarak ELM'ye gönderiyoruz.
OrnekX=pi/2;
NetGirisCikarim = W * OrnekX;
Hcikarim = 1 ./ (1 + exp(-NetGirisCikarim));        
OrnekY = Beta * Hcikarim;
%disp(["Sin(x) tahmini için, ",OrnekX, "şeklinde bir girişe karşılık elde edilen sonuç: ", OrnekY, "olmuştur"])
fprintf('\nSin(x) tahmini için, %4.2f (%2.1f pi) olarak verilen bir giriş için hesaplanan çıktı %4.2f olmuştur\n',OrnekX,OrnekX/pi,OrnekY)
fprintf('Sin(%4.2f) için gerçek değer ise %4.2f olmalıdır\n',OrnekX, sin(OrnekX))
fprintf('Eğitilen sistem, %%%4.2f hata yapmıştır.\n\n',abs(100*(sin(OrnekX)-OrnekY)))

%% Grafikler
subplot(3,1,3); plot(X_test,Y_tahmin, 'r.'); hold on; plot(X_test,Y_test, 'b+'); title('MSEArray')
legend('Tahmin Y','Gerçek Y')
