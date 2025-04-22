# Mini GPT-2 (char-level) in PyTorch

Implementazione minimale di GPT-2 in PyTorch, per scopi didattici. Include attenzione causale, MLP e blocchi Transformer. Funziona a livello di caratteri.

## Chatbot nel Notebook

Questo progetto implementa un modello GPT-2 a livello di carattere in PyTorch, che può essere utilizzato come chatbot in un notebook interattivo. L'utente può scrivere un input e il modello risponde generando un testo di continuità basato sull'input fornito.

### Tecniche di Ottimizzazione Utilizzate

Durante il processo di addestramento e inferenza sono stati implementati vari miglioramenti per ottimizzare il modello e le prestazioni:

1. **Precisione Mista (Mixed Precision)**:
   La precisione mista è stata utilizzata durante il training e l'inferenza per accelerare i calcoli e ridurre l'utilizzo della memoria. PyTorch supporta la precisione mista tramite `torch.cuda.amp.autocast()`, che aiuta a velocizzare l'elaborazione senza compromettere troppo la qualità dei risultati.

2. **Clip Gradienti**:
   Durante il training, il clipping dei gradienti è stato applicato per evitare il problema dei gradienti esplosivi. Questo è particolarmente utile per i modelli con molteplici parametri, come i Transformer.

3. **Scheduling del Tasso di Apprendimento (Learning Rate Scheduling)**:
   Un scheduler per il tasso di apprendimento è stato utilizzato per ridurre il tasso di apprendimento durante il training, migliorando la stabilità e l'efficacia dell'ottimizzazione.

4. **Warm-up del Tasso di Apprendimento**:
   Un periodo di warm-up per il tasso di apprendimento è stato implementato all'inizio del training per permettere al modello di adattarsi gradualmente, migliorando la convergenza nelle prime fasi del training.

5. **Early Stopping con Salvataggio del Miglior Modello**:
   L'addestramento si ferma se non si osserva un miglioramento nelle metriche di validazione, e il modello con la miglior performance viene salvato per evitare l'overfitting.

### Utilizzo del Chatbot

Il chatbot è implementato nel notebook come segue:

1. **Esegui il Codice di Setup**: Inizializza il modello, le variabili e i dizionari necessari per la tokenizzazione.
2. **Inserisci un Messaggio**: L'utente può digitare una domanda o una frase nel campo di input.
3. **Generazione della Risposta**: Il modello genererà una risposta basata sull'input dell'utente.
4. **Risposte Iterative**: Il chatbot continua a rispondere alle nuove domande finché non viene interrotto.