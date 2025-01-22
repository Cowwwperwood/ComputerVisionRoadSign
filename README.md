# Классификация редких дорожных знаков

Этот проект посвящен классификации редких дорожных знаков. Основная проблема классификации дорожных знаков заключается в следующем: датасет некоторых дорожных знаков найти проще, чем других. Например : изображения, на которых есть знак "Пешеходный переход" не составляет труда, но в то же время найти изображение со знаком "Габариты автомобиля" найти является настолько сложной задачей, что наш датасет , при ручном поиске, может ограничиться парой - десятком изображений, что для обучения нейронной сети является критически недостаточным. В данном проекте мы будем симуляровать отсутствие редких знаков и использовать их будем только для тестирования.

## Структура проекта

Проект состоит из трех основных частей:

1. **Pre-training нейронной сети на изображениях включающих только частые классы**:
   - На первом этапе была загружена предварительно обученная нейросеть ResNet50 , заменили последний линейный слой, после чего, в ходе эксперементов мы выяснили то, что эффективнее всего(лучшие метрики) были достигнуты при отсутствии заморозки слоев. На данном этапе мы получили качество: 85% accuracy на частых дорожных знаках(на тех классах, на которых мы учились) и , соответственно,  0% на редких дорожных знаках.
2. **Генерация синтетических данных**:
   - На втором этапе мы взяли изображения всех дорожных знаков с прозрачным фоном (маской знака в 4 цветовом канале) и 1000 изображений заднего фона, после чего сгенерировали синтетические данные, путем: каждый дорожный знак мы обработали(сделали более тусклым , заблюрили, повернули на случайный угол, подробнее с классом генерации можно ознакомиться в директории Tools), наложили задний фон, и сгенерировали 1000 синтетических изображений каждого класса. В итоге мы получили 205_000 новых изображенйи. Стоит заметить, что в моей реализации с базовыми параметрами функций аугментаций некоторые изображения оказались слишком плохого качества(слишком маленький размер изображений, слишком сильное размытие), что подпортило результаты обучения. Я это , к сожалению, заметил слишком поздно.

3. **Тренировка на синтетических данных**:
   - На третьем этапе мы взяли предобученную модель из первого пункта, провели эксперименты с обучением, а именно: полное обучение модели(все веса разморожены), частичное обучение(заморожены все слои , кроме предпоследнего блока ResNet, активации(ReLU) и линейного слоя), а также предобученную на ImageNet ResNet50, лучшие результаты показала модель , частично обучавшаяся. Метрики: Accuracy на всех знаках(и редких и частых) : 67.7.
   ![Test Acc](https://github.com/Cowwwperwood/ComputerVisionRoadSign/blob/main/image/TestStat.png)
   - Accuracy только на редких: 52.1.
   ![Rare Acc](https://github.com/Cowwwperwood/ComputerVisionRoadSign/blob/main/image/RareStat.png)

   - Результаты можно улучшить(примерно до 80/65), путем смешивания синтетических данных и реальных, а также более качественной генерации синтетических данных.

4. **Тренировка на синтетических данных**:
   - Далее была предпринята попытка воспользоваться методами метрического обучения, путем: написания лосса-добавки(а именно contrastive loss) к кросс-энтропии(ознакомиться с ним можно в Tools), небольшим редактированием модели(помимо логитов мы возвращаем еще и вектор-признаки с предпоследнего слоя), а также написан BatchSampler. Дело в том, что классов 205, а в выборке они представлены неравномерно. Поэтому какие-то классы будут семплироваться часто, какие-то реже, в одном батче большинство объектов будут разного класса.
Мы же хотим контролировать, чтобы внутри одного батча было classes_per_batch разных классов и для каждого класса было ровно elems_per_class объектов (например, classes_per_batch = 32 и elems_per_class = 4). Тогда наша loss-функция будет иметь смысл. Приступив к обучению на миксе из синтетических данных и реальных я столкнулся с тем, что обучение одной эпохи будет занимать 6 часов(вследствие того, что увеличился датасет, прибавилась необходимость считать два лосса, а также, самая главная причина - BatchSampler, скорее всего моя реализация с точки зрения эффективности не оптимальная). В связи с эти было решено прекратить обучение из-за недостатка вычислительных мощностей для этого обучения.
   ![Rare Acc](https://github.com/Cowwwperwood/ComputerVisionRoadSign/blob/main/image/FailMet.png)


## Заключение
Проект продемонстрировал возможность применения синтетических данных для классификации редких дорожных знаков. Несмотря на ограниченные вычислительные ресурсы, удалось достичь значительных результатов. Тем не менее, качество синтетических данных и подход к обучению требуют дальнейшей оптимизации. Для улучшения метрик рекомендуется:
   - Увеличить количество реальных данных;
   - Улучшить генерацию синтетических изображений;
   - Оптимизировать реализацию BatchSampler для эффективного обучения.
   - Проект показал, что даже при недостатке данных можно добиться значительного прогресса в задачах классификации.


