# Oppgave 2

## 2 a, b)
Funnksjonene virker som de skal, vi satte opp et par matriser i $\mathbb{C}^{2x2}$ med unike elementer og satte dem som argumenter til $f : \mathbb{C}^{2x2} \rightarrow \mathbb{R}^8$ og verifisert at det som kom ut var på formen $\left[ a_1 ,a_2,a_3,a_4,b_1,b_2,b_3,b_4\right]$. Gjorde det samme med genrelle vektorer i $\mathbb{R}^8$ i oppgave B.

## 2g)
Vi løser systemet med begge grenseveriene lik null og med begge gjettningene lik null. Dermed vil likning 9-12 alle ende med deriverte lik null. Dette gjelder også for likning 13-16. Derfor forventer vi at alle løsningsvektorene blir null. Og dermed at tilstandstettheten vil bli konstant. Dette stemmer, *solve_bvp* returnerer alle løsningsvektorer lik $\vec{0}$. Dette gir mening ettersom det ikke er noen mulighet for superledende egenskaper å lekke inn i det normale metallet.

## 2j)
I figur 1 ser vi at tilstandstettheten for $E - E_F = 2 |\Delta| \iff \varepsilon = 2$ er noe større enn 1. Dermed er det flere tilgjengelige tilstander for disse elektronene i superledende fase enn i normal fase for det superledende materiale. På grunn av Proximity effect (herfra referert til som nærhets effekten) vil denne økningen i antall tilstander "lekke" inn i det normale metallet. Dette gjør at kantene på normal fase metall vil ha høyere tilstandstetthet enn ved sentrum i $x = \frac{l}{2}$. Der er det å forvente at resultatene vil være likere de i oppgave h, med tilnærmet kontant verdi. 

Når vi plotter dette ser vi at tilstandstettheten er noe lavere i det normale metallet når det grenser til superledere enn når det ikke gjør det. Den forutsatte formen på den plottede kurven stemmer, men ved vår forståelse burde punktene $x \in \{ 0, l\}$ ha tilstandstetthet noe større enn 1. Gitt tilstanden i superlederene ved denne energien.

## 2k)
Det superledende gapet dukker opp for mindre energier. Vi ser også at etterhvert som lengden av den normale fasen øker minker mengden energier som inngår i gapet. Dette skyldes at nærhets effekten avtar etterhvert som avstanden mellom superlederene øker. Om vi skal gjette tipper vi det skyldes at sannsynligheten for tunnelering synker . 

Videre ser vi at for $\varepsilon > 1$ jevner tilstandstettheten ut og går mot den normale fasen, dette likner på det vi ser i fig 1. Og er å forvente ettersom vi når energier som er større enn Fermi energien, hvilket lar elektroner nå de lendene båndene. Materialet får da normale ledeevner, hvilket stemmer overens med oppg j.


## 2l

Som vist i plottet er strøm-integranden lik null for alle energiene. Dersom man ikke koder inn grenser for y-aksen vil det vise variasjoner på størrelsesordenen $10^{-15}$, men ettersom toleransen på solveren til scipy er satt til $10^{-6}$ er dette et resultat av begrensninger i numerisk beregning og ikke et fysisk resultat.

## 2m

Nå som $\Delta\phi\neq0$ får vi som forventet en strøm i metallet, i motsetning til oppgave 2l. Dersom vi istedenfor energi plotter strømmen som funksjon av posisjon for fem utvalgte verdier av epsilon slik som i oppgave 2l ser vi likevel at strøm-integranden også her er konservert, ettersom den er konstant for alle x, dette oppfyller kravet $j(x,\varepsilon) = j(0,\varepsilon)$.

**HER KAN DET GJØRES ET ARGUMENT FOR AT DEN ER KONSTANT MED MASSE-, LADNING- OG ENERGIBEVARING, ER DETTE NOE VI VIL HA MED?**

## 2n

Vi observerer at strømmen øker for økende $\Delta\phi$ inntil et maksimum ved $\Delta\phi = \frac{\pi}{2}$, og den synker deretter til 0 ved $\Delta\phi=\pi$. Deretter blir den negativ men symetrisk om y-aksen. Slik at den kan tilnærmes med en sinus funksjon. Strømmen er altså periodisk i $\Delta\phi$ med perioden p = $2\pi$.

Videre observerer vi at den sinusodiale formen til $J(\Delta\phi)$ har noen overtoner. Disse har ikke store utslag og blir kun tydelige når vi har høy oppløsning langs $\Delta \phi$-aksen. Vi vet ikke om dette er resultat av at vi bruker færre verdier av $\varepsilon$ enn det som vanligvis gjøres (ref *a* i oppgaveteksten) eller om dette er et faktisk fysisk fenomen som kan overserveres.



