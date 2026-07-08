"""
Generatore definitivo del dataset per il fine-tuning (Estrazione Sottoricette).
Produce esempi REALI E UNICI in formato ChatML (.jsonl).

Include Hard Negatives per addestrare il modello a ignorare preparazioni semplici.
"""

import json
import random

# Fissiamo il seed per la riproducibilità (Best Practice Accademica)
random.seed(42)

SYSTEM_PROMPT = """Sei un estrattore specializzato di sottoricette per un blog di cucina italiana.

Il tuo compito è leggere una ricetta e identificare SOLO le sottoricette autonome, cioè preparazioni che:
1. Richiedono COTTURA, EMULSIONE o MONTATURA per esistere (es. besciamella, ragù, maionese, crema pasticcera)
2. Hanno una propria identità culinaria riconoscibile al di fuori di questa ricetta
3. Potrebbero essere preparate e usate indipendentemente

NON sono sottoricette:
- Marinature o miscele di spezie crude (es. "marinatura al limone", "mix di spezie")
- Condimenti a crudo (es. olio e limone, salsa di soia con zenzero)
- Panature (farina + uovo + pangrattato)
- Ingredienti pronti o semilavorati (pesto in barattolo, brodo di dado)
- Semplici tagli o preparazioni banali (trito di cipolla, aglio schiacciato)

Rispondi ESCLUSIVAMENTE con un array JSON valido di stringhe che utilizzi il seguente formato:
    -Se non ci sono sottoricette,  DEVI rispondere SOLO ED ESCLUSIVAMENTE con [] SENZA NESSUNA ARGOMENTAZIONE. 
    -Se ci sono sottoricette, DEVI rispondere SOLO ED ESCLUSIVAMENTE con un array JSON di stringhe contenente i nomi delle sottoricette identificate, SENZA NESSUNA ARGOMENTAZIONE.
    -Esempi validi: ["Besciamella"], ["Ragù", "Besciamella"] nel caso di sottoricette trovate, oppure [] nel caso in cui non ci siano sottoricette.

Il tuo output deve essere un array JSON valido di stringhe, SONO VIETATI commenti o spiegazioni aggiuntive.
"""

# Dataset pulito, validato e formattato correttamente
ESEMPI = [
    # --- RICETTE CON NESSUNA SOTTORICETTA (O DOVE LA RICETTA STESSA E' LA BASE) ---
    (
        "Pollo marinato alla griglia\nIngredienti: petto di pollo, olio d'oliva, limone, rosmarino, sale. \nProcedimento: Marinate il pollo per 2 ore con olio, limone e rosmarino. Grigliate e servite.",
        [],
    ),
    (
        "Insalata di polpo\nIngredienti: polpo, patate, sedano, olio, limone, prezzemolo. \nProcedimento: Lessate polpo e patate. Tagliate a pezzi e condite a crudo con olio e limone.",
        [],
    ),
    (
        "Spaghetti aglio, olio e peperoncino\nIngredienti: spaghetti, aglio, olio d'oliva, peperoncino. \nProcedimento: Cuocete gli spaghetti. Fate dorare l'aglio con olio e peperoncino. Saltate la pasta.",
        [],
    ),
    (
        "Frittata di zucchine\nIngredienti: uova, zucchine, cipolla, olio, sale, parmigiano. \nProcedimento: Stufate le zucchine. Sbattete le uova con parmigiano, unite le zucchine e cuocete in padella.",
        [],
    ),
    (
        "Carpaccio di manzo\nIngredienti: filetto di manzo, rucola, grana, olio, limone. \nProcedimento: Disponete la carne cruda, coprite con rucola e grana. Condite a crudo.",
        [],
    ),
    (
        "Sashimi di salmone\nIngredienti: salmone fresco, salsa di soia, wasabi. \nProcedimento: Affettate il salmone. Servite crudo con salsa di soia già pronta.",
        [],
    ),
    (
        "Bruschetta al pomodoro\nIngredienti: pane, pomodori maturi, basilico, aglio, olio. \nProcedimento: Grigliate il pane. Condite i pomodori crudi con basilico e olio. Distribuite sul pane.",
        [],
    ),
    (
        "Gazpacho andaluso\nIngredienti: pomodori, cetriolo, peperone, cipolla, aglio, olio, aceto. \nProcedimento: Frullate tutti gli ingredienti crudi. Filtrate e tenete in frigo.",
        [],
    ),
    (
        "Caprese\nIngredienti: mozzarella di bufala, pomodori, basilico, olio, sale. \nProcedimento: Tagliate mozzarella e pomodori a fette. Condite a crudo.",
        [],
    ),
    (
        "Guacamole\nIngredienti: avocado, pomodoro, cipolla, coriandolo, lime. \nProcedimento: Schiacciate l'avocado crudo e unire con gli altri ingredienti",
        [],
    ),
    (
        "Tzatziki\nIngredienti: yogurt greco, cetriolo, aglio, aneto. \nProcedimento: Mescolate yogurt con cetriolo grattugiato e aglio crudo",
        [],
    ),
    (
        "Panzanella\nIngredienti: pane raffermo, pomodori, cetriolo, cipolla, basilico, aceto. \nProcedimento: Bagnate il pane. Condite a crudo con le verdure tagliate, olio e aceto.",
        [],
    ),
    (
        "Tartare di manzo\nIngredienti: manzo, capperi, senape, limone, tuorlo d'uovo. \nProcedimento: Tritate la carne al coltello. Condite a crudo e adagiatevi un tuorlo crudo.",
        [],
    ),
    (
        "Poke bowl\nIngredienti: riso, salmone, avocado, edamame, salsa di soia. \nProcedimento: Cuocete il riso. Condite il salmone crudo con soia. Componete la bowl.",
        [],
    ),
    (
        "Avocado toast\nIngredienti: pane, avocado, pomodorini, limone, semi di chia. \nProcedimento: Tostate il pane. Schiacciate l'avocado crudo con limone e spalmate.",
        [],
    ),
    (
        "Smoothie tropicale\nIngredienti: mango, banana, latte di cocco, granola. \nProcedimento: Frullate la frutta cruda con latte di cocco.",
        [],
    ),
    (
        "Cozze alla marinara\nIngredienti: cozze, aglio, olio, vino bianco, prezzemolo. \nProcedimento: Fate aprire le cozze in padella con aglio, olio e vino. Servite.",
        [],
    ),
    (
        "Peperoni arrostiti\nIngredienti: peperoni, aglio, olio, prezzemolo. \nProcedimento: Abbrustolite i peperoni, spellateli e condite a crudo con aglio e olio.",
        [],
    ),
    (
        "Melanzane al funghetto\nIngredienti: melanzane, pomodorini, aglio, olio. \nProcedimento: Friggete le melanzane a cubetti. Saltatele con pomodorini freschi.",
        [],
    ),
    (
        "Zuppa di miso\nIngredienti: dashi granulare, miso, tofu, alga wakame. \nProcedimento: Sciogliete dashi e miso in acqua calda. Aggiungete tofu e alghe.",
        [],
    ),
    (
        "Udon saltati con verdure\nIngredienti: udon precotti, peperoni rossi, zucchine,peperoni gialli, carote, cipolotto fresco, aglio, salsa di soia, olio di semi, peperoncino, semi d sesamo. \nProcedimento: Saltate le verdure. Aggiungete udon precotti e salsa di soia.",
        [],
    ),
    (
        "Curry di lenticchie\nIngredienti: lenticchie, cipolla, pomodori, curry in polvere, latte di cocco. \nProcedimento: Cuocete le lenticchie con le verdure. Aggiungete spezie e latte di cocco.",
        [],
    ),
    (
        "Hummus\nIngredienti: ceci in barattolo, tahina, limone, aglio. \nProcedimento: Frullate i ceci precotti con tahina e aglio crudo.",
        [],
    ),
    (
        "Crostini con paté di olive\nIngredienti: olive nere, capperi, acciughe, aglio, pane. \nProcedimento: Frullate a crudo le olive con capperi e acciughe. Spalmate sul pane.",
        [],
    ),
    (
        "Stracciatella alla romana\nIngredienti: uova, parmigiano, brodo di carne in brick. \nProcedimento: Portate a bollore il brodo pronto, versatevi le uova sbattute.",
        [],
    ),
    (
        "Pasta e fagioli\nIngredienti: pasta, fagioli in scatola, pomodori, aglio. \nProcedimento: Soffriggete aglio, aggiungete pomodori e fagioli precotti. Cuocete la pasta dentro.",
        [],
    ),
    (
        "Spaghetti alle vongole\nIngredienti: spaghetti, vongole, aglio, vino bianco. \nProcedimento: Aprite le vongole in padella, saltatevi gli spaghetti cotti.",
        [],
    ),
    (
        "Insalata di riso\nIngredienti: riso, tonno, olive, mais, wurstel. \nProcedimento: Cuocete il riso. Mescolate a freddo con tutti gli ingredienti.",
        [],
    ),
    (
        "Banana bread\nIngredienti: banane, farina, uova, lievito, noci. \nProcedimento: Mescolate tutto in una ciotola (nessuna precottura) e infornate.",
        [],
    ),
    (
        "Muffin al cioccolato\nIngredienti: farina, cacao, uova, latte, lievito. \nProcedimento: Mescolate ingredienti secchi e umidi a freddo. Infornate.",
        [],
    ),
    (
        "Impasto per pancakes\nIngredienti: farina, latte, uova, lievito, burro. \nProcedimento: Mescolate tutto a crudo e cuocete in padella a mestoli.",
        [],
    ),
    (
        "French toast\nIngredienti: pancarrè, uova, latte, burro. \nProcedimento: Immergete il pane nelle uova crude e cuocete in padella.",
        [],
    ),
    (
        "Shakshuka\nIngredienti: uova, pomodori pelati, cipolla, peperone, spezie. \nProcedimento: Stufate i pomodori pelati con le spezie, rompetevi le uova crude dentro e cuocete.",
        [],
    ),
    (
        "Carbonara vegetariana\nIngredienti: spaghetti, zucchine, uova, pecorino. \nProcedimento: Saltate le zucchine. Mantecate a crudo con la crema di uova e formaggio.",
        [],
    ),
    (
        "Pasta cacio e pepe\nIngredienti: tonnarelli, pecorino, pepe. \nProcedimento: Cuocete la pasta. Create una crema a crudo con pecorino, pepe e acqua.",
        [],
    ),
    (
        "Prosciutto e melone\nIngredienti: prosciutto crudo, melone cantalupo. \nProcedimento: Tagliate il melone, avvolgetelo nel prosciutto. Nessuna cottura.",
        [],
    ),
    (
        "Bresaola rucola e grana\nIngredienti: bresaola, rucola, scaglie di grana, limone. \nProcedimento: Disponete la bresaola, coprite con rucola e grana. Condite a crudo.",
        [],
    ),
    (
        "Spaghetti al tonno\nIngredienti: spaghetti, tonno , capperi, olive. \nProcedimento: Scolate la pasta e saltate velocemente in padella col tonno.",
        [],
    ),
    (
        "Macedonia di frutta\nIngredienti: mele, pere, banane, fragole, succo di limone, zucchero. \nProcedimento: Tagliate la frutta a cubetti, mescolate a crudo con limone e zucchero.",
        [],
    ),
    (
        "Insalata di quinoa\nIngredienti: quinoa, pomodorini, cetriolo, feta, olio, limone. \nProcedimento: Cuocete la quinoa. Mescolate a crudo con gli altri ingredienti.",
        [],
    ),
    (
        "Cozze gratinate\nIngredienti: cozze, pangrattato, aglio, prezzemolo, olio. \nProcedimento: Aprite le cozze, riempitele con pangrattato e aglio crudo. Gratinate in forno.",
        [],
    ),
    (
        "Insalata di farro\nIngredienti: farro, pomodorini, rucola, feta, olio, limone. \nProcedimento: Cuocete il farro. Mescolate a crudo con gli altri ingredienti.",
        [],
    ),
    (
        "Cous cous di verdure\nIngredienti: cous cous, zucchine, peperoni, pomodorini, olio, limone. \nProcedimento: Cuocete il cous cous. Saltate le verdure e mescolate a crudo con il cous cous.",
        [],
    ),
    (
        "Polpo alla gallega\nIngredienti: polpo, patate, paprika, olio, sale. \nProcedimento: Lessate il polpo e le patate. Tagliate a pezzi e condite a crudo con paprika e olio.",
        [],
    ),
    (
        "Orata al forno\nIngredienti: orata, limone, prezzemolo, olio, sale. \nProcedimento: Pulite l'orata e conditela a crudo con limone e prezzemolo. Infornate.",
        [],
    ),
    (
        "Cotoletta alla milanese\nIngredienti: fettine di vitello, uova, pangrattato, burro, limone. \nProcedimento: Passate le fettine crude nell'uovo e pangrattato. Friggete in burro e servite con limone.",
        [],
    ),
    (
        "Maionese\nIngredienti: uova, olio di semi, senape, limone, sale. \nProcedimento: Frullate le uova crude con olio e limone fino a ottenere una crema. (E' la ricetta madre)",
        [],
    ),
    (
        "Pesto alla genovese\nIngredienti: basilico, pinoli, aglio, parmigiano, olio d'oliva. \nProcedimento: Frullate a crudo tutti gli ingredienti fino a ottenere una crema.",
        [],
    ),
    (
        "Pesto al basilico e noci\nIngredienti: basilico, noci, aglio, parmigiano, olio d'oliva. \nProcedimento: Frullate a crudo tutti gli ingredienti fino a ottenere una crema.",
        [],
    ),
    (
        "Crema pasticcera\nIngredienti: latte, uova, zucchero, farina, vaniglia. \nProcedimento: Scaldate il latte. Mescolate a crudo uova, zucchero e farina. Aggiungete il latte caldo e cuocete fino a ottenere una crema.",
        [],
    ),
    (
        "Tiramisù\nIngredienti: savoiardi, mascarpone, uova, zucchero, caffè. \nProcedimento: Montate a crudo tuorli e zucchero. Aggiungete mascarpone. Inzuppate i savoiardi nel caffè e alternate con crema.",
        [],
    ),
    (
        "Linguine al nero di seppia\nIngredienti: pasta, nero di seppia, aglio, olio, prezzemolo. \nProcedimento: Cuocete la pasta. Saltate con aglio e olio e aggiungete il nero di seppia a crudo.",
        [],
    ),
    (
        "Besciamella\nIngredienti: burro, farina, latte, noce moscata, sale. \nProcedimento: Sciogliete il burro. Aggiungete farina e mescolate a crudo. Versate il latte caldo e cuocete fino a ottenere una crema.",
        [],
    ),
    (
        "Salsa di pomodoro\nIngredienti: pomodori pelati, aglio, olio, basilico, sale. \nProcedimento: Frullate i pomodori crudi con aglio e olio. Cuocete fino a ottenere una salsa.",
        [],
    ),
    (
        "Ragù alla bolognese\nIngredienti: carne macinata, pomodori, cipolla, carota, sedano, vino rosso. \nProcedimento: Soffriggete le verdure. Aggiungete la carne cruda e rosolate. Sfumate con vino e aggiungete pomodori. Cuocete a fuoco lento.",
        [],
    ),
    (
        "Pasta biscotto\nIngredienti: uova, zucchero, farina, lievito. \nProcedimento: Montate a crudo uova e zucchero. Aggiungete farina e lievito. Infornate fino a doratura.",
        [],
    ),
    (
        "Pasta frolla\nIngredienti: farina, burro, zucchero, uova, vaniglia. \nProcedimento: Mescolate a crudo farina e burro. Aggiungete zucchero e uova. Formate un impasto e refrigerate.",
        [],
    ),
    (
        "Pasta sfoglia\nIngredienti: farina, burro, acqua, sale. \nProcedimento: Impastate a crudo farina e acqua. Incorporate il burro freddo e lavorate fino a ottenere una sfoglia. Refrigerate.",
        [],
    ),
    (
        "Pasta brisée\nIngredienti: farina, burro, acqua, sale. \nProcedimento: Impastate a crudo farina e acqua. Incorporate il burro freddo e lavorate fino a ottenere una pasta. Refrigerate.",
        [],
    ),
    (
        "Ganache al cioccolato\nIngredienti: cioccolato fondente, panna, burro\nProcedimento: Scaldate la panna. Versatela sul cioccolato tritato a crudo e mescolate fino a ottenere una crema liscia. Aggiungete burro.",
        [],
    ),
    (
        "Crema chantilly\nIngredienti: panna, zucchero a velo, vaniglia. \nProcedimento: Montate a crudo la panna con zucchero e vaniglia fino a ottenere una crema soffice.",
        [],
    ),
    (
        "Crema al burro\nIngredienti: burro, zucchero a velo, uova, vaniglia. \nProcedimento: Montate a crudo il burro con zucchero e uova fino a ottenere una crema liscia. Aggiungete vaniglia.",
        [],
    ),
    (
        "Ragù siciliano\nIngredienti: carne di maiale, pomodori, cipolla, aglio, vino rosso. \nProcedimento: Soffriggete cipolla e aglio. Aggiungete la carne cruda e rosolate. Sfumate con vino e aggiungete pomodori. Cuocete a fuoco lento.",
        [],
    ),
    (
        "Pollo al curry\nIngredienti: pollo, cipolla, aglio, curry in polvere, latte di cocco. \nProcedimento: Soffriggete cipolla e aglio. Aggiungete il pollo crudo e rosolate. Aggiungete curry e latte di cocco. Cuocete fino a cottura.",
        [],
    ),
    (
        "Pasta fresca all'uovo\nIngredienti: farina, uova, sale. \nProcedimento: Impastate a crudo farina e uova fino a ottenere una pasta liscia. Stendete e tagliate a piacere.",
        [],
    ),
    (
        "Gnocchi di patate\nIngredienti: patate, farina, uova, sale. \nProcedimento: Lessate le patate. Schiacciatele a crudo e mescolate con farina e uova fino a ottenere un impasto. Formate gli gnocchi.",
        [],
    ),
    (
        "Mousse al cioccolato\nIngredienti: cioccolato fondente, uova, zucchero, panna. \nProcedimento: Sciogliete il cioccolato. Montate a crudo gli albumi con zucchero e la panna. Incorporate il cioccolato fuso e mescolate delicatamente.",
        [],
    ),
    (
        "Crema catalana\nIngredienti: latte, zucchero, tuorli d'uovo, amido di mais, cannella, scorza di limone. \nProcedimento: Scaldate il latte con cannella e scorza di limone. Mescolate a crudo tuorli, zucchero e amido. Aggiungete il latte caldo e cuocete fino a ottenere una crema. Raffreddate e caramellate lo zucchero in superficie.",
        [],
    ),
    (
        "Brodo vegetale\nIngredienti: carote, sedano, cipolla, porro, acqua, sale. \nProcedimento: Tagliate le verdure a pezzi. Mettetele in una pentola con acqua e sale. Cuocete fino a ottenere un brodo saporito.",
        [],
    ),
    (
        "Brodo di carne\nIngredienti: ossa di manzo, carote, sedano, cipolla, acqua, sale. \nProcedimento: Mettete le ossa e le verdure in una pentola con acqua e sale. Cuocete a fuoco lento per diverse ore fino a ottenere un brodo ricco.",
        [],
    ),
    (
        "Brodo di pesce\nIngredienti: teste e lische di pesce, carote, sedano, cipolla, acqua, sale. \nProcedimento: Mettete le teste e lische di pesce con le verdure in una pentola con acqua e sale. Cuocete a fuoco lento per circa 30-40 minuti. Filtrate il brodo.",
        [],
    ),
    (
        "Salsa bearnaise\nIngredienti: burro, aceto, scalogno, dragoncello, tuorli d'uovo, sale. \nProcedimento: Riducete a crudo aceto e scalogno. Montate i tuorli a bagnomaria. Aggiungete il burro fuso e il dragoncello. Cuocete fino a ottenere una salsa liscia.",
        [],
    ),
    (
        "Salsa olandese\nIngredienti: burro, tuorli d'uovo, succo di limone, sale. \nProcedimento: Montate i tuorli a bagnomaria. Aggiungete il burro fuso e il succo di limone. Cuocete fino a ottenere una salsa liscia.",
        [],
    ),
    (
        "Bruschetta con crema di avocado\nIngredienti: pane, avocado, limone, sale, pepe. \nProcedimento: Tostate il pane. Schiacciate l'avocado crudo con limone, sale e pepe. Spalmate sul pane.",
        [],
    ),
    (
        "Impasto per crepes\nIngredienti: farina, uova, latte, zucchero. \nProcedimento: Mescolate a crudo farina, uova, latte e zucchero fino a ottenere una pastella. Cuocete le crepes in padella.",
        [],
    ),
    (
        "Impasto per pizza\nIngredienti: farina, acqua, lievito, sale, olio d'oliva. \nProcedimento: Mescolate a crudo farina, acqua e lievito. Aggiungete sale e olio. Impastate fino a ottenere un impasto elastico. Lasciate lievitare.",
        [],
    ),
    (
        "Polpette di carne\nIngredienti: carne macinata, uova, pangrattato, parmigiano, prezzemolo, sale. \nProcedimento: Mescolate a crudo carne, uova, pangrattato, parmigiano e prezzemolo. Formate le polpette e friggete.",
        [],
    ),
    (
        "Pasta per crespelle\nIngredienti: farina, uova, latte, burro. \nProcedimento: Mescolate a crudo farina, uova e latte fino a ottenere una pastella. Cuocete le crespelle in padella con burro.",
        [],
    ),
    (
        "Purè di patate\nIngredienti: patate, burro, latte, noce moscata, sale. \nProcedimento: Lessate le patate. Schiacciatele e mescolatele con burro e latte bollente fino a ottenere un purè cremoso.",
        [],
    ),
    (
        "Ragù di cinghiale\nIngredienti: carne di cinghiale, cipolla, carota, sedano, vino rosso, pomodori pelati. \nProcedimento: Soffriggete le verdure. Aggiungete la carne cruda e rosolate. Sfumate con vino e aggiungete pomodori. Cuocete a fuoco lento.",
        [],
    ),
    (
        "Pasta choux\nIngredienti: acqua, burro, farina, uova.\nProcedimento: Portate a bollore acqua e burro. Aggiungete la farina e mescolate a crudo. Incorporate le uova una alla volta.",
        [],
    ),
    (
        "Polenta\nIngredienti: farina di mais, acqua, sale.\nProcedimento: Portate a bollore l'acqua salata. Aggiungete la farina di mais a pioggia e mescolate fino a ottenere una polenta densa.",
        [],
    ),
    (
        "Cialda per cannoli\nIngredienti: farina 00, zucchero, uova, strutto, vino bianco, lievito, olio di semi. \nProcedimento: Preparate l’impasto per i cannoli, avvolgetelo su cilindri di metallo e friggetelo. Farcite solo al momento con ricotta zuccherata cruda.",
        [],
    ),
    (
        "Pan di spagna\nIngredienti: uova, zucchero, farina, lievito. \nProcedimento: Montate a crudo uova e zucchero. Aggiungete farina e lievito. Infornate fino a doratura.",
        [],
    ),
    (
        "Salsa teriyaki\nIngredienti: salsa di soia, zucchero, sake, mirin. \nProcedimento: Mescolate a crudo tutti gli ingredienti e scaldate fino a ottenere una salsa densa.",
        [],
    ),
    (
        "Cheesecake\nIngredienti: biscotti, burro, formaggio cremoso, zucchero, uova, panna. \nProcedimento: Preparate la base con biscotti e burro fuso. Montate a crudo formaggio, zucchero e uova. Versate sulla base e infornate.",
        [],
    ),
    (
        "Pastella\nIngredienti: farina, acqua frizzante, uova, sale. \nProcedimento: Mescolate a crudo farina, acqua frizzante e uova fino a ottenere una pastella liscia. Usatela per friggere.",
        [],
    ),
    (
        "Pasta per lasagne\nIngredienti: farina 00, farina di grano duro, uova, sale. \nProcedimento: Impastate a crudo farina e uova fino a ottenere una pasta liscia. Stendete e tagliate a strisce per lasagne.",
        [],
    ),
    (
        "Ragù bianco\nIngredienti: carne macinata, cipolla, carota, sedano, vino bianco, brodo. \nProcedimento: Soffriggete le verdure. Aggiungete la carne cruda e rosolate. Sfumate con vino e aggiungete brodo. Cuocete a fuoco lento.",
        [],
    ),
    (
        "Ragù napoletano\nIngredienti: carne di manzo, carne di maiale, pomodori pelati, cipolla, vino rosso. \nProcedimento: Soffriggete la cipolla. Aggiungete le carni crude e rosolate. Sfumate con vino e aggiungete pomodori. Cuocete a fuoco lento.",
        [],
    ),
    (
        "Crema di riso\nIngredienti: riso, latte, zucchero, vaniglia. \nProcedimento: Cuocete il riso nel latte con zucchero e vaniglia fino a ottenere una crema densa.",
        [],
    ),
    # ---  RICETTE CON UNA SOTTORICETTA ---
    (
        "Moussaka\nIngredienti: melanzane, carne bovina macinata, cipolle dorate, latte intero, burro, farina 00.\nProcedimento: Preparate la besciamella fondendo burro e farina e versando il latte a filo. Friggete le melanzane. Rosolate la carne. Stratificate in forno.",
        ["Besciamella"],
    ),
    (
        "Cannelloni ripieni\nIngredienti: cannelloni secchi, ricotta vaccina, spinaci freschi, latte intero, burro, farina 00, noce moscata in polvere.\nProcedimento: Preparate la besciamella classica cuocendola sul fuoco. Farcite i cannelloni con ricotta e spinaci. Coprite di besciamella e infornate.",
        ["Besciamella"],
    ),
    (
        "Pasticcio di maccheroni\nIngredienti: maccheroni, latte intero, burro, farina 00, prosciutto cotto.\nProcedimento: Preparate una besciamella densa. Lessate la pasta, condite con la crema ottenuta e il prosciutto, e infornate.",
        ["Besciamella"],
    ),
    (
        "Sformato di zucchine\nIngredienti: zucchine, uova, latte intero, burro, farina 00.\nProcedimento: Preparate una besciamella media. Mescolate con zucchine saltate e uova. Infornate.",
        ["Besciamella"],
    ),
    (
        "Crostata di marmellata\nIngredienti: farina 00, burro, zucchero semolato, uova, confettura di albicocche.\nProcedimento: Preparate la pasta frolla impastando velocemente burro freddo, farina e uova. Foderate la tortiera, farcite di marmellata e infornate.",
        ["Pasta frolla"],
    ),
    (
        "Biscotti da tè\nIngredienti: farina 00, burro, uova, zucchero semolato, limoni non trattati.\nProcedimento: Preparate la pasta frolla impastando le polveri con burro, uova e scorza di limone. Fatela riposare, tagliate i biscotti e infornate.",
        ["Pasta frolla"],
    ),
    (
        "Quiche Lorraine\nIngredienti: farina 00, burro, acqua, pancetta affumicata, uova, panna fresca liquida.\nProcedimento: Preparate la pasta brisée lavorando farina, burro e acqua. Stendetela in teglia, versate il ripieno liquido e cuocete.",
        ["Pasta brisée"],
    ),
    (
        "Torta salata alle verdure\nIngredienti: farina 00, burro, acqua, ricotta vaccina, zucchine.\nProcedimento: Impastate la brisée a freddo e lasciatela in frigo. Stendetela e versatevi il mix di ricotta cruda e zucchine saltate.",
        ["Pasta brisée"],
    ),
    (
        "Ravioli ricotta e spinaci\nIngredienti: farina 00, uova, ricotta vaccina, spinaci freschi, burro.\nProcedimento: Impastate farina e uova per la pasta fresca all'uovo. Farcite con il ripieno crudo, cuocete e condite con burro fuso.",
        ["Pasta fresca all'uovo"],
    ),
    (
        "Tagliatelle al pomodoro\nIngredienti: farina 00, uova, pomodori pelati, aglio.\nProcedimento: Tirate a mano la pasta fresca all'uovo impastata precedentemente. Conditela con un sugo di pomodoro basico.",
        ["Pasta fresca all'uovo"],
    ),
    (
        "Gnocchi di patate al pomodoro\nIngredienti: patate a pasta gialla, farina 00, uova, passata di pomodoro.\nProcedimento: Preparate gli gnocchi lessando le patate, schiacciandole e impastandole con uovo e farina. Condite con passata scaldata.",
        ["Gnocchi di patate"],
    ),
    (
        "Gnocchi alla sorrentina\nIngredienti: patate a pasta gialla, farina 00, uova, mozzarella fiordilatte, passata di pomodoro.\nProcedimento: Preparate gli gnocchi a mano lessando i tuberi. Cuoceteli e ripassateli in forno con mozzarella e pomodoro.",
        ["Gnocchi di patate"],
    ),
    (
        "Pizza Margherita\nIngredienti: farina di grano tenero tipo 0, acqua, lievito di birra fresco, sale fino, passata di pomodoro, mozzarella fiordilatte.\nProcedimento: Preparate l'impasto lievitato per pizza. Stendetelo, condite e infornate a 250°.",
        ["Impasto per pizza"],
    ),
    (
        "Focaccia ligure\nIngredienti: farina 00, acqua, lievito di birra fresco, sale grosso, olio extravergine d'oliva.\nProcedimento: Preparate l'impasto per focaccia ad alta idratazione. Mettetelo in teglia, fate i buchi, condite e infornate.",
        ["Impasto per focaccia"],
    ),
    (
        "Zuppa di pesce\nIngredienti: pesce da zuppa, teste di pesce, lische di pesce, sedano, carote, cipolle dorate, aglio, pomodori pelati.\nProcedimento: Preparate il brodo di pesce bollendo gli scarti con gli odori. Soffriggete aglio, unite i pesci e il brodo filtrato.",
        ["Brodo di pesce"],
    ),
    (
        "Risotto alla pescatora\nIngredienti: riso Carnaroli, cozze, calamari, scarti di pesce, sedano, carote, cipolle dorate.\nProcedimento: Preparate il fumet (brodo di pesce). Tostate il riso e portatelo a cottura col brodo bollente e il pesce in padella.",
        ["Brodo di pesce"],
    ),
    (
        "Risotto giallo\nIngredienti: riso Carnaroli, zafferano in pistilli, carote, sedano, cipolle dorate, acqua.\nProcedimento: Bollite sedano carota e cipolla per fare il brodo vegetale. Usatelo bollente per cuocere il risotto.",
        ["Brodo vegetale"],
    ),
    (
        "Polenta pasticciata\nIngredienti: farina di mais fioretto, acqua, sale grosso, funghi porcini freschi.\nProcedimento: Cuocete la polenta nel paiolo per 40 minuti versando la farina a pioggia. Servitela coperta dai funghi saltati.",
        ["Polenta"],
    ),
    (
        "Polenta concia\nIngredienti: farina di mais bramata, acqua, sale grosso, Fontina DOP, burro.\nProcedimento: Cuocete la polenta classica girando sempre sul fuoco. Mantecatela alla fine con i formaggi.",
        ["Polenta"],
    ),
    (
        "Panna cotta\nIngredienti: panna fresca liquida, gelatina in fogli, zucchero semolato, acqua.\nProcedimento: Preparate il caramello fondendo zucchero e acqua in un pentolino. Versatevi sopra la panna cotta e fate rassodare in frigo.",
        ["Caramello"],
    ),
    (
        "Torta rovesciata all'ananas\nIngredienti: ananas fresco, zucchero semolato, burro, farina 00, uova, lievito in polvere per dolci.\nProcedimento: Fate il caramello in padella fondendo lo zucchero. Versateci l'impasto crudo della torta e l'ananas, infornate.",
        ["Caramello"],
    ),
    (
        "Pollo al caramello salato\nIngredienti: petto di pollo, zucchero semolato, acqua, salsa di soia.\nProcedimento: Sciogliete lo zucchero in padella per fare il caramello e allungatelo con salsa di soia. Cuocetevi il pollo a pezzi.",
        ["Caramello"],
    ),
    (
        "Spaghetti al ragù\nIngredienti: spaghetti, carne bovina macinata, carne suina macinata, sedano, carote, cipolle dorate, passata di pomodoro, vino rosso.\nProcedimento: Preparate il vero ragù bolognese rosolando le carni e stufandole per 3 ore. Condite la pasta secca.",
        ["Ragù bolognese"],
    ),
    # RUMORE 1: Questa la lasciamo con il semilavorato esplicito per far generalizzare il modello
    (
        "Spaghetti al ragù di cinghiale\nIngredienti: pappardelle all'uovo, ragù di cinghiale .\nProcedimento: Fate stufare la carne tritata per il ragù di cinghiale. Usatelo per la pasta in busta.",
        ["Ragù di cinghiale"],
    ),
    (
        "Parmigiana classica\nIngredienti: melanzane ovali nere, pomodori pelati, aglio, basilico fresco, olio di semi di arachide, mozzarella fiordilatte.\nProcedimento: Preparate il sugo di pomodoro ristretto in casseruola. Alternate le melanzane fritte col sugo e infornate.",
        ["Sugo di pomodoro"],
    ),
    (
        "Polpette al sugo\nIngredienti: carne bovina macinata, uova, mollica di pane, passata di pomodoro, aglio, basilico fresco.\nProcedimento: Preparate il sugo di pomodoro in tegame. Tuffatevi le polpette crude e fatele cuocere nel liquido bollente.",
        ["Sugo di pomodoro"],
    ),
    # RUMORE 2: Lasciamo con "sugo di pomodoro" esplicito negli ingredienti
    (
        "Uova al purgatorio\nIngredienti: uova, sugo di pomodoro, peperoncino rosso piccante.\nProcedimento: Fate restringere il sugo di pomodoro in padella. Rompetevi dentro le uova e coprite.",
        ["Sugo di pomodoro"],
    ),
    (
        "Tarte Tatin\nIngredienti: farina 00, burro, acqua, mele renette, zucchero semolato.\nProcedimento: Impastate la brisée a freddo. Caramellate le mele e copritele col disco di pasta prima di infornare.",
        ["Pasta brisée"],
    ),
    (
        "Crostata ricotta e visciole\nIngredienti: farina 00, burro, uova, zucchero semolato, ricotta di pecora, visciole sciroppate.\nProcedimento: Fate la pasta frolla. Riempite il guscio steso con ricotta zuccherata e visciole dal barattolo.",
        ["Pasta frolla"],
    ),
    (
        "Millefoglie alla panna\nIngredienti: farina 00, acqua, burro, panna fresca liquida.\nProcedimento: Dedicatevi ai giri e alle pieghe per ottenere la pasta sfoglia. Cuocetela e farcite solo con panna montata.",
        ["Pasta sfoglia"],
    ),
    # RUMORE 3
    (
        "Vol-au-vent ai funghi\nIngredienti: pasta sfoglia, funghi champignon.\nProcedimento: Create i cestini di pasta sfoglia e infornateli. Riempiteli con funghi cotti in padella.",
        ["Pasta sfoglia"],
    ),
    (
        "Bignè alla panna\nIngredienti: acqua, burro, farina 00, uova, panna fresca liquida.\nProcedimento: Preparate la pasta choux sul fuoco, poi infornate i bignè finché non sono gonfi. Riempiteli di semplice panna.",
        ["Pasta choux"],
    ),
    (
        "Eclair al cioccolato (ripieni di panna)\nIngredienti: acqua, burro, farina 00, uova, panna fresca liquida, cioccolato fondente.\nProcedimento: Formate i bastoncini di pasta choux e infornate. Farciteli con panna e intingeteli nel cioccolato fuso.",
        ["Pasta choux"],
    ),
    (
        "Crema catalana caramellata\nIngredienti: latte intero, uova, amido di mais, zucchero semolato, zucchero di canna.\nProcedimento: Cuocete il liquido per la crema. Al momento di servire, fondete lo zucchero in superficie per creare la crosta di caramello.",
        ["Caramello"],
    ),
    (
        "Crepe alla Nutella\nIngredienti: farina 00, uova, latte intero, burro, Nutella.\nProcedimento: Preparate la pastella liquida e cuocete le crepes in padella versando un mestolo alla volta. Farcitele con crema spalmabile pronta.",
        ["Crepes"],
    ),
    # RUMORE 4
    (
        "Torta salata\nIngredienti: pasta sfoglia, spinaci freschi, ricotta vaccina.\nProcedimento: Impastate la pasta sfoglia con burro e farina. Versatevi il mix di ricotta a crudo e infornate.",
        ["Pasta sfoglia"],
    ),
    (
        "Zuppa inglese rapida\nIngredienti: savoiardi, latte intero, tuorli, zucchero semolato, amido di mais, baccello di vaniglia.\nProcedimento: Preparate la crema pasticcera profumata alla vaniglia addensandola sul fuoco. Alternate con i biscotti bagnati.",
        ["Crema pasticcera"],
    ),
    (
        "Salmone glassato\nIngredienti: filetto di salmone, salsa di soia, mirin, sake, zucchero semolato.\nProcedimento: Riducete in pentolino gli ingredienti liquidi con lo zucchero per creare una salsa teriyaki densa. Spennellate il pesce in cottura.",
        ["Salsa teriyaki"],
    ),
    (
        "Risotto ai funghi porcini\nIngredienti: riso Carnaroli, funghi porcini freschi, cipolle dorate, carote, sedano, acqua, vino bianco, Parmigiano Reggiano DOP.\nProcedimento: Preparate il brodo vegetale facendo sobbollire le verdure in acqua. Tostate il riso, sfumate col vino e portate a cottura aggiungendo il brodo caldo e i funghi.",
        ["Brodo vegetale"],
    ),
    (
        "Passatelli in brodo\nIngredienti: pangrattato, Parmigiano Reggiano DOP, uova, noce moscata in polvere, carne bovina, carote, sedano, cipolle dorate. \nProcedimento: Preparate il brodo di carne bollendo gli scarti e le verdure. Mescolate pangrattato, formaggio e uova e pressateli nello schiacciapatate direttamente nel brodo bollente.",
        ["Brodo di carne"],
    ),
    (
        "Tortellini burro e salvia\nIngredienti: farina 00, uova, lombo di maiale, mortadella, burro, salvia fresca. \nProcedimento: Impastate le polveri con uova per la pasta fresca e tirate una sfoglia. Create un ripieno crudo con le carni, formate i tortellini. Cuoceteli e saltateli con burro fuso e salvia.",
        ["Pasta fresca all'uovo"],
    ),
    (
        "Cappellacci di zucca al burro\nIngredienti: farina 00, uova, zucca mantovana, amaretti, Parmigiano Reggiano DOP, burro, salvia fresca. \nProcedimento: Impastate la sfoglia fresca. Schiacciate la zucca cotta e mescolatela con amaretti e formaggio. Farcite i cappellacci, lessateli e conditeli con burro e salvia.",
        ["Pasta fresca all'uovo"],
    ),
    (
        "Cannoli siciliani\nIngredienti: farina 00, strutto, zucchero semolato, vino Marsala, cacao amaro in polvere, ricotta di pecora, zucchero a velo. \nProcedimento: Preparate l’impasto per la cialda per cannoli, avvolgetelo su cilindri e friggetelo. Farcite con ricotta zuccherata a crudo.",
        ["Cialda per cannoli"],
    ),
    (
        "Cassata siciliana\nIngredienti: farina 00, uova, zucchero semolato, ricotta di pecora, gocce di cioccolato fondente, frutta candita mista. \nProcedimento: Preparate e cuocete il pan di spagna. Lavorate la ricotta con lo zucchero. Foderate uno stampo con fette di torta soffice e farcite con la crema di ricotta.",
        ["Pan di spagna"],
    ),
    (
        "Bruschetta con caponata\nIngredienti: melanzane, pomodori ramati, sedano, olive verdi, capperi sotto sale, aceto di vino bianco, zucchero semolato, pane casereccio. \nProcedimento: Preparate la caponata stufando tutte le verdure in agrodolce in padella. Lasciate intiepidire e servite su fette di pane bruscato.",
        ["Caponata"],
    ),
    (
        "Pasta alla Norma\nIngredienti: maccheroni, melanzane, pomodori pelati, aglio, olio extravergine d'oliva, basilico fresco, ricotta salata. \nProcedimento: Preparate un sugo di pomodoro semplice cuocendolo con aglio e basilico. Friggete le melanzane a cubetti. Cuocete la pasta e conditela con il sugo e le melanzane.",
        ["Sugo di pomodoro"],
    ),
    (
        "Tortelli di erbette al burro\nIngredienti: farina 00, uova, bietole, ricotta vaccina, Parmigiano Reggiano DOP, burro. \nProcedimento: Preparate la sfoglia all'uovo a mano. Lessate le bietole, strizzatele e mescolatele con ricotta e parmigiano. Farcite i tortelli, cuoceteli e conditeli con burro fuso.",
        ["Pasta fresca all'uovo"],
    ),
    (
        "Parmigiana di zucchine\nIngredienti: zucchine chiare, passata di pomodoro, cipolle bianche, olio extravergine d'oliva, mozzarella fiordilatte, Parmigiano Reggiano DOP, basilico fresco. \nProcedimento: Preparate il sugo di pomodoro sul fuoco. Grigliate le zucchine a fette. Alternate in teglia zucchine, sugo e formaggi, poi gratinate in forno.",
        ["Sugo di pomodoro"],
    ),
    (
        "Timballo di maccheroni\nIngredienti: maccheroni, latte intero, burro, farina 00, noce moscata in polvere, prosciutto cotto, provola, pangrattato. \nProcedimento: Preparate la besciamella classica in pentolino. Lessate i maccheroni, conditeli con la salsa bianca e il prosciutto, versateli in uno stampo imburrato e spolverate di pangrattato. Gratinate in forno.",
        ["Besciamella"],
    ),
    (
        "Arancini al ragù\nIngredienti: riso Vialone Nano, carne bovina macinata, carne suina macinata, cipolle dorate, pisellini, passata di pomodoro, caciocavallo, pangrattato, uova. \nProcedimento: Preparate il ragù siciliano facendolo ritirare molto. Lessate il riso e conditelo. Formate delle palline e friggetele.",
        ["Ragù siciliano"],
    ),
    (
        "Insalata russa\nIngredienti: patate a pasta gialla, carote, pisellini, uova, olio di semi di girasole, succo di limone. \nProcedimento: Lessate le verdure. Preparate la maionese frullando tuorli crudi, olio e limone. Mescolate verdure e uova sode a pezzetti con l'emulsione.",
        ["Maionese"],
    ),
    (
        "Spezzatino di carne\nIngredienti: carne bovina, cipolle dorate, vino rosso, patate a pasta gialla, burro, latte intero.\nProcedimento: Stufate la carne con la cipolla. A parte preparate il purè di patate schiacciando i tuberi lessi e unendo burro e latte. Servite lo spezzatino sulla crema.",
        ["Purè di patate"],
    ),
    (
        "Insalata di pollo\nIngredienti: petto di pollo, tuorli, olio di semi di girasole, succo di limone, sedano, carote, uova. \nProcedimento: Lessate il pollo. Preparate la maionese in casa emulsionando tuorli crudi e olio. Mescolate il pollo con la salsa e il sedano.",
        ["Maionese"],
    ),
    (
        "Uova alla diavola\nIngredienti: uova, olio di semi di girasole, succo di limone, paprika dolce, senape di Digione.\nProcedimento: Tagliate le uova sode. Preparate la maionese montando olio e tuorli. Mescolate la polpa soda delle uova con l'emulsione, paprika e senape.",
        ["Maionese"],
    ),
    (
        "Salsa rosa\nIngredienti: tuorli, olio di semi di girasole, succo di limone, ketchup, brandy. \nProcedimento: Preparate la maionese fatta in casa montandola con un frullatore. Mescolate poi l'emulsione con ketchup e brandy fino a ottenere una salsa colorata.",
        ["Maionese"],
    ),
    (
        "Pizza capricciosa\nIngredienti: farina di grano tenero tipo 0, acqua, lievito di birra fresco, sale fino, passata di pomodoro, mozzarella fiordilatte, prosciutto cotto, funghi champignon, olive nere, carciofini sott'olio. \nProcedimento: Preparate l'impasto per pizza impastando a lungo le polveri con l'acqua. Stendetelo e conditelo con pomodoro, mozzarella e gli altri ingredienti. Infornate.",
        ["Impasto per pizza"],
    ),
    (
        "Pizza quattro stagioni\nIngredienti: farina di grano tenero tipo 0, acqua, lievito di birra fresco, olio extravergine d'oliva, passata di pomodoro, mozzarella fiordilatte, prosciutto cotto, funghi champignon, carciofini sott'olio, olive nere. \nProcedimento: Preparate l'impasto per pizza. Stendetelo in teglia e conditelo dividendo a spicchi pomodoro, mozzarella e gli altri ingredienti. Infornate.",
        ["Impasto per pizza"],
    ),
    (
        "Supplì\nIngredienti: riso Carnaroli, mozzarella fiordilatte, pangrattato, uova, passata di pomodoro, cipolle dorate, olio di semi di arachide. \nProcedimento: Preparate il sugo di pomodoro stufando cipolla e passata. Lessate il riso e conditelo con il sugo. Formate delle palline, inserite un cubetto di mozzarella, passatele nell'uovo e nel pangrattato e friggetele.",
        ["Sugo di pomodoro"],
    ),
    (
        "Fiori di zucca fritti\nIngredienti: fiori di zucca, mozzarella fiordilatte, acciughe sott'olio, farina 00, acqua frizzante, olio di semi di arachide. \nProcedimento: Preparate la pastella mescolando velocemente farina e acqua con la frusta. Farcite i fiori con mozzarella e acciughe crude. Passateli nel liquido e friggeteli.",
        ["Pastella"],
    ),
    # ---  RICETTE CON DUE O PIU' SOTTORICETTE ---
    (
        "Lasagne alla bolognese\nIngredienti: farina 00, uova, carne bovina macinata, carne suina macinata, sedano, carote, cipolle dorate, passata di pomodoro, latte intero, burro.\nProcedimento: Preparate il ragù lento di carne cuocendolo tre ore. A parte cuocete la besciamella al burro sul fuoco. Tirate la sfoglia. Assemblate a strati.",
        ["Ragù alla bolognese", "Besciamella", "Pasta fresca all'uovo"],
    ),
    (
        "Lasagne vegetariane\nIngredienti: pasta all'uovo, zucchine, melanzane, peperoni rossi, latte intero, burro, farina 00.\nProcedimento: Preparate la besciamella addensandola sul fuoco. Grigliate le verdure. Alternate strati in teglia con la pasta pronta.",
        ["Pasta per lasagne", "Besciamella"],
    ),
    (
        "Cannelloni alla bolognese\nIngredienti: cannelloni secchi, carne bovina macinata, sedano, carote, passata di pomodoro, latte intero, burro, farina 00, noce moscata in polvere.\nProcedimento: Fate il ragù alla bolognese facendolo stringere bene. Fate la besciamella. Riempite i tubi di pasta, coprite con le salse e infornate.",
        ["Ragù alla bolognese", "Besciamella"],
    ),
    (
        "Pasticcio al forno\nIngredienti: maccheroni, carne bovina macinata, carne suina macinata, sedano, carote, cipolle dorate, vino bianco, latte intero, burro, farina 00.\nProcedimento: Preparate il ragù bianco di macinato senza pomodoro sfumando col vino. Fate la besciamella e mischiate tutto con la pasta sbollentata.",
        ["Ragù bianco", "Besciamella"],
    ),
    (
        "Timballo di anelletti\nIngredienti: anelletti, carne bovina macinata, pisellini, concentrato di pomodoro, latte intero, burro, farina 00.\nProcedimento: Cuocete il ragù siciliano con i piselli e addensate la besciamella in pentolino. Condite la pasta e versate in uno stampo a ciambella.",
        ["Ragù siciliano", "Besciamella"],
    ),
    (
        "Crespelle fiorentine\nIngredienti: farina 00, uova, latte intero, ricotta vaccina, spinaci freschi, burro.\nProcedimento: Cuocete le crespelle versando il liquido a mestoli in padella. Farcite con ricotta, arrotolate e coprite di besciamella fatta in pentola.",
        ["Crespelle", "Besciamella"],
    ),
    (
        "Crostata della domenica\nIngredienti: farina 00, burro, uova, zucchero semolato, latte intero, tuorli, amido di mais, frutti di bosco.\nProcedimento: Cuocete il guscio di pasta frolla in bianco impastando a mano. Riempitelo di crema pasticcera addensata a parte sul fuoco.",
        ["Pasta frolla", "Crema pasticcera"],
    ),
    (
        "Torta della nonna\nIngredienti: farina 00, burro, uova, zucchero semolato, latte intero, tuorli, amido di mais, scorza di limone, pinoli sgusciati.\nProcedimento: Fate la pasta frolla. Fate la crema pasticcera densa girandola sul fuoco. Farcite un disco di pasta, chiudete con un altro disco, coprite di pinoli e infornate.",
        ["Pasta frolla", "Crema pasticcera"],
    ),
    (
        "Bignè con crema pasticcera\nIngredienti: acqua, burro, farina 00, uova, latte intero, tuorli, zucchero semolato, amido di mais.\nProcedimento: Cuocete l'impasto di pasta bignè in pentola poi infornatelo a ciuffetti fino a che gonfia. Una volta freddi, siringateli con crema pasticcera fatta sul fuoco.",
        ["Pasta bignè", "Crema pasticcera"],
    ),
    # RUMORE 5
    (
        "Profiteroles classici\nIngredienti: pasta choux, crema pasticcera, cioccolato fondente.\nProcedimento: Formate la pasta choux e cuocetela. Riempiteli di crema pasticcera. La glassa è solo cioccolato fuso.",
        ["Pasta choux", "Crema pasticcera"],
    ),
    (
        "Millefoglie tradizionale\nIngredienti: farina 00, acqua, burro, latte intero, tuorli, zucchero semolato, baccello di vaniglia.\nProcedimento: Eseguite i giri e le pieghe della pasta sfoglia e infornatela bucherellata. Farcite gli strati croccanti con la crema pasticcera preparata a caldo.",
        ["Pasta sfoglia", "Crema pasticcera"],
    ),
    (
        "Torta diplomatica\nIngredienti: farina 00, burro, acqua, uova, zucchero semolato, latte intero, tuorli, amido di mais, rum scuro.\nProcedimento: Eseguite le pieghe per i fogli croccanti. Montate le uova per cuocere un pan di spagna alto. Preparate la crema pasticcera e assemblate il tutto bagnando col liquore.",
        ["Pasta sfoglia", "Pan di spagna", "Crema pasticcera"],
    ),
    (
        "Tortellini di carne\nIngredienti: farina 00, uova, carne bovina, carne di gallina, carne suina macinata, noce moscata in polvere, sedano, carote. \nProcedimento: Bollite carni miste e verdure per ore per fare il brodo di carne. Impastate le polveri, tirate la sfoglia, chiudete i tortellini e lessateli nel liquido bollente.",
        ["Pasta fresca all'uovo", "Brodo di carne"],
    ),
    (
        "Pasticcio di polenta rustico\nIngredienti: farina di mais, acqua, sale grosso, salsiccia di suino, passata di pomodoro, cipolle dorate.\nProcedimento: Cuocete la polenta nel paiolo. Stendetela a fette e usatela a strati con il ragù di salsiccia stufato nel pomodoro.",
        ["Polenta", "Ragù di salsiccia"],
    ),
    (
        "Pizza napoletana\nIngredienti: farina di grano tenero tipo 00, acqua, lievito di birra fresco, carne bovina, passata di pomodoro, cipolle dorate.\nProcedimento: Lievitate l'impasto per pizza. Preparate il classico ragù napoletano a lenta cottura (pappuliando). Farcite due dischi sovrapposti e infornate.",
        ["Impasto per pizza", "Ragù napoletano"],
    ),
    (
        "Calzoni fritti al ragù\nIngredienti: farina 00, acqua, lievito di birra fresco, carne bovina macinata, pomodori pelati, cipolle dorate, olio di semi di arachide.\nProcedimento: Fate l'impasto per pizza. Riempitelo con un cucchiaio di ragù napoletano ristretto cotto per ore in anticipo e friggete a immersione.",
        ["Impasto per pizza", "Ragù napoletano"],
    ),
    (
        "Sartù di riso base\nIngredienti: riso Carnaroli, carne bovina, cipolle dorate, passata di pomodoro, latte intero, burro, farina 00.\nProcedimento: Lessate il riso. Assemblatelo in uno stampo ricoprendo il cuore con ragù napoletano e besciamella addensata sul fuoco.",
        ["Ragù napoletano", "Besciamella"],
    ),
    (
        "Gateau di patate al forno\nIngredienti: patate a pasta gialla, latte intero, burro, carne bovina macinata, sedano, carote, passata di pomodoro, farina 00, uova.\nProcedimento: Preparate il purè di patate sodo. Farcite in teglia con strati di besciamella fatta a parte e ragù alla bolognese.",
        ["Purè di patate", "Ragù alla bolognese", "Besciamella"],
    ),
    (
        "Quiche saporita\nIngredienti: farina 00, burro, acqua, latte intero, porri.\nProcedimento: Foderate lo stampo con la pasta brisée fresca lavorata a mano. Mescolate i porri saltati con la besciamella preparata nel pentolino e infornate.",
        ["Pasta brisée", "Besciamella"],
    ),
    (
        "Crostata di frutta mista\nIngredienti: farina 00, burro, zucchero semolato, uova, latte intero, tuorli, baccello di vaniglia, pesche, kiwi.\nProcedimento: Impastate e cuocete il guscio di pasta frolla. Riempitelo di crema pasticcera addensata a parte sul fuoco e decorate.",
        ["Pasta frolla", "Crema pasticcera"],
    ),
    (
        "Bignè caramellati\nIngredienti: acqua, burro, farina 00, uova, latte intero, tuorli, zucchero semolato.\nProcedimento: L'impasto di pasta bignè viene cotto in forno a palline, farcito con crema pasticcera fatta sul fuoco e tuffato nel caramello bollente per formare la crosta dura.",
        ["Pasta bignè", "Crema pasticcera", "Caramello"],
    ),
    (
        "Eclair alla vaniglia\nIngredienti: acqua, burro, farina 00, uova, latte intero, tuorli, zucchero semolato, baccello di vaniglia.\nProcedimento: Fate i classici eclair siringando la pasta choux, tagliateli a metà e riempiteli di crema pasticcera addensata ai semi di vaniglia.",
        ["Pasta choux", "Crema pasticcera alla vaniglia"],
    ),
    (
        "Torta rustica di mele\nIngredienti: farina 00, burro, zucchero semolato, uova, latte intero, tuorli, amido di mais, mele renette.\nProcedimento: Impastate e stendete la pasta frolla nello stampo. Adagiatevi uno strato di crema pasticcera cotta in pentolino e affondatevi le fette di mela cruda.",
        ["Pasta frolla", "Crema pasticcera"],
    ),
    (
        "Torta salata panna e funghi\nIngredienti: farina 00, burro, acqua, latte intero, funghi champignon, uova.\nProcedimento: Usate la pasta brisée cruda come base. Copritela con funghi trifolati annegati in una besciamella leggera e uova.",
        ["Pasta brisée", "Besciamella"],
    ),
    (
        "Torta di riso classica\nIngredienti: farina 00, burro, uova, zucchero semolato, latte intero, riso Originario, baccello di vaniglia.\nProcedimento: Stendete la base di pasta frolla. Preparate la crema di riso cuocendo i chicchi nei liquidi zuccherati e versatela nel guscio. Infornate.",
        ["Pasta frolla", "Crema di riso"],
    ),
    (
        "Millefoglie di polenta\nIngredienti: farina di mais fioretto, acqua, sale grosso, carne bovina macinata, sedano, carote, passata di pomodoro, latte intero, burro, farina 00.\nProcedimento: Preparate la polenta sul fuoco e fatela raffreddare a dischi. Intervallatela con ragù alla bolognese e salsa besciamella.",
        ["Polenta", "Ragù alla bolognese", "Besciamella"],
    ),
    (
        "Bignè salati\nIngredienti: acqua, burro, farina 00, uova, latte intero, Parmigiano Reggiano DOP.\nProcedimento: Formate i bignè di pasta choux sul fuoco e infornate. Riempiteli con besciamella calda arricchita di parmigiano e servite.",
        ["Pasta choux", "Besciamella"],
    ),
    (
        "Gateau di patate napoletano\nIngredienti: patate a pasta gialla, latte intero, burro, carne bovina, cipolle dorate, pomodori pelati, farina 00, mozzarella fiordilatte.\nProcedimento: Preparate il purè schiacciando le patate col burro. Preparate il ragù napoletano lunghissimo. Preparate la besciamella. In teglia alternate strati e infornate a 180°C.",
        ["Purè di patate", "Ragù napoletano", "Besciamella"],
    ),
    (
        "Pasticcio di polenta con ragù e besciamella\nIngredienti: farina di mais bramata, acqua, sale grosso, carne bovina macinata, passata di pomodoro, cipolle dorate, latte intero, burro, farina 00, Parmigiano Reggiano DOP\nProcedimento: Cuocete la polenta nel paiolo. Preparate il ragù di carne e addensate la besciamella. In una pirofila alternate fette di mais, sugo e salsa, gratinando in forno.",
        ["Polenta", "Ragù di carne", "Besciamella"],
    ),
    (
        "Lasagne di carnevale\nIngredienti: farina 00, uova, carne bovina macinata, mollica di pane, passata di pomodoro, cipolle dorate, latte intero, burro.\nProcedimento: Preparate le polpette e cuocetele. Stufate il sugo di pomodoro. Fate la besciamella. Impastate e lessate la pasta per lasagne, conditela e alternate in teglia.",
        ["Pasta per lasagne", "Polpette di carne", "Sugo di pomodoro", "Besciamella"],
    ),
    (
        "Timballo di riso alla siciliana\nIngredienti: riso Carnaroli, carne bovina macinata, pisellini, estratto di pomodoro, latte intero, burro, farina 00, Parmigiano Reggiano DOP.\nProcedimento: Cuocete il riso. Preparate il ragù siciliano e la besciamella a parte. In uno stampo alternate strati di riso, sugo e salsa. Infornate.",
        ["Ragù siciliano", "Besciamella"],
    ),
    # RUMORE 6
    (
        "Torta ai setteveli\nIngredienti: cioccolato fondente, crêpes dentelles, pasta biscotto, bavarese alla vaniglia, bavarese alla nocciola, mousse al cioccolato, zucchero semolato, panna fresca liquida, gelatina in fogli, cacao amaro in polvere. \nProcedimento: Preparate la base di pasta biscotto. Stratificate sopra la bavarese semplice e poi la bavarese alla nocciola. Coprite tutto con la mousse al cioccolato e glassate a specchio.",
        [
            "Pasta biscotto",
            "Bavarese",
            "Bavarese alla nocciola",
            "Mousse al cioccolato",
        ],
    ),
]


def format_chatml(testo, labels):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Analizza questa ricetta e identifica le sottoricette presenti:\n\n{testo}",
            },
            {"role": "assistant", "content": json.dumps(labels, ensure_ascii=False)},
        ]
    }


def genera_e_salva_dataset():
    print(f"Generazione in corso... Totale esempi forniti: {len(ESEMPI)}")

    dataset_formattato = [format_chatml(t, l) for t, l in ESEMPI]

    # Stratifichiamo per bilanciare esempi "vuoti" (nessuna sottoricetta) e "non-vuoti"
    vuoti = [
        ex
        for ex in dataset_formattato
        if json.loads(ex["messages"][2]["content"]) == []
    ]
    non_vuoti = [
        ex
        for ex in dataset_formattato
        if json.loads(ex["messages"][2]["content"]) != []
    ]

    random.shuffle(vuoti)
    random.shuffle(non_vuoti)

    def split_lista(lista):
        n_train = int(len(lista) * 0.8)
        n_val = int(len(lista) * 0.1)
        return (
            lista[:n_train],
            lista[n_train : n_train + n_val],
            lista[n_train + n_val :],
        )

    train_v, val_v, test_v = split_lista(vuoti)
    train_n, val_n, test_n = split_lista(non_vuoti)

    train = train_v + train_n
    val = val_v + val_n
    test = test_v + test_n

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    splits = {"train": train, "val": val, "test": test}
    for nome, dati in splits.items():
        with open(f"sottoricette_{nome}.jsonl", "w", encoding="utf-8") as f:
            for ex in dati:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open("sottoricette_dataset.jsonl", "w", encoding="utf-8") as f:
        for ex in dataset_formattato:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("✅ Dataset stratificato generato con successo!")
    print(f"📊 Train: {len(train)} (vuoti={len(train_v)}, non_vuoti={len(train_n)})")
    print(f"📊 Val:   {len(val)} (vuoti={len(val_v)}, non_vuoti={len(val_n)})")
    print(f"📊 Test:  {len(test)} (vuoti={len(test_v)}, non_vuoti={len(test_n)})")


if __name__ == "__main__":
    genera_e_salva_dataset()
