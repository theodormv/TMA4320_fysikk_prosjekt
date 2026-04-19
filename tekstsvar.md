## Oppgave 2

### 2 a, b)
Funnksjonene virker som de skal, vi satte opp et par matriser i $\mathbb{C}^{2x2}$ med unike elementer og satte dem som argumenter til $f : \mathbb{C}^{2x2} \rightarrow \mathbb{R}^8$ og verifisert at det som kom ut var på formen $\left[ a_1 ,a_2,a_3,a_4,b_1,b_2,b_3,b_4\right]$. Gjorde det samme med genrelle vektorer i $\mathbb{R}^8$ i oppgave B.

### 2g)
Vi løser systemet med begge grenseveriene lik null og med begge gjettningene lik null. Dermed vil likning 9-12 alle ende med deriverte lik null. Dette gjelder også for likning 13-16. Derfor forventer vi at alle løsningsvektorene blir null. Og dermed at tilstandstettheten vil bli konstant. Dette stemmer, *solve_bvp* returnerer alle løsningsvektorer lik $\vec{0}$. Dette gir mening ettersom det ikke er noen mulighet for superledende egenskaper å lekke inn i det normale metallet.

### 2j)
I figur 1 ser vi at tilstandstettheten for $E - E_F = 2 |\Delta| \iff \epsilon = 2$ er noe større enn 1. Dermed er det flere tilgjengelige tilstander for disse elektronene i superledende fase enn i normal fase for det superledende materiale. På grunn av Proximity effect (herfra referert til som nærhets effekten) vil denne økningen i antall tilstander "lekke" inn i det normale metallet. Dette gjør at kantene på normal fase metall vil ha høyere tilstandstetthet enn ved sentrum i $x = \frac{l}{2}$. Der er det å forvente at resultatene vil være likere de i oppgave h, med tilnærmet kontant verdi. 

Når vi plotter dette ser vi at tilstandstettheten er noe lavere i det normale metallet når det grenser til superledere enn når det ikke gjør det. Den forutsatte formen på den plottede kurven stemmer, men ved vår forståelse burde punktene $x \in \{ 0, l\}$ ha tilstandstetthet noe større enn 1. Gitt tilstanden i superlederene ved denne energien.

### 2k)
Det superledende gapet dukker opp for mindre energier. Vi ser også at etterhvert som lengden av den normale fasen øker minker mengden energier som inngår i gapet. Dette skyldes at nærhets effekten avtar etterhvert som avstanden mellom superlederene øker. Om vi skal gjette tipper vi det skyldes at sannsynligheten for tunnelering synker . 

Videre ser vi at for $\epsilon > 1$ jevner tilstandstettheten ut og går mot den normale fasen, dette likner på det vi ser i fig 1. Og er å forvente ettersom vi når energier som er større enn Fermi energien, hvilket lar elektroner nå de lendene båndene. Materialet får da normale ledeevner, hvilket stemmer overens med oppg j.
