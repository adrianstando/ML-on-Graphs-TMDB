# ML-on-Graphs-TMDB
The project created during the Machine Learning on Graphs course.


### Project description 

> in Polish

Naszym pomysłem na projekt jest stworzenie modelu regresji do predykcji ‘revenue’ filmów na podstawie datasetu https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata. Jako baseline — metodę bez użycia grafów — chcielibyśmy użyć MLP i/lub innych modeli ML  wytrenowanych na bag of words z opisów filmów/keywordsach dotyczących danego filmu i innych zmiennych opisujących produkcje.

Pierwszym proponowanym sposobem inferencji grafu jest przedstawienie  każdego filmu jako wierzchołka i dodanie krawędzi między filmami, które mają duże przecięcie obsady/ekipy filmowej. Krawędzie dodawalibyśmy, gdy podobieństwo mierzone np. indeksem Jaccarda byłoby powyżej pewnego thresholdu. Wagą krawędzi byłoby właśnie to podobieństwo.

Drugim pomysłem na inferencję grafu byłoby uwzględnienie keywordsów opisujących film również w tym procesie, tzn. krawędzie między filmami byłyby wyznaczane na podstawie ich podobieństwa.

Ewentualnym trzecim pomysłem jest przygotowanie grafu o różnym typie wierzchołków, gdzie obok filmów wierzchołki reprezentowałyby także aktorów i ekipę filmową — filmy nie byłyby połączone krawędziami między sobą, ale właśnie z tymi wierzchołkami.

W każdym z przypadków node feature’y byłyby takie same jak w podejściu niegrafowym.