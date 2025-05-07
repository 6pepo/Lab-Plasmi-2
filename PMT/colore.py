import cv2
import numpy as np

def show_fullscreen_color(vals):
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
    except ImportError:
        print("Avviso: 'tkinter' non trovato. Usando dimensioni di default (1024x768).")
        print("Installa tkinter (es. 'pip install tk') per un rilevamento migliore delle dimensioni dello schermo.")
        screen_width = 1024
        screen_height = 768
    except Exception as e:
        print(f"Errore nel rilevare le dimensioni dello schermo con tkinter: {e}")
        print("Usando dimensioni di default (1024x768).")
        screen_width = 1024
        screen_height = 768


    # Crea un'immagine nera con le dimensioni dello schermo (height, width, channels)
    # OpenCV usa il formato BGR (Blue, Green, Red)
    image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    for i in range(3):
        image[:,:,i] = vals[i]

    # Crea una finestra senza bordi e a schermo intero
    cv2.namedWindow("Fullscreen Color", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Fullscreen Color", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Mostra l'immagine
    cv2.imshow("Fullscreen Color", image)

    # Attendi la pressione di un tasto e poi chiudi la finestra
    cv2.waitKey(0)
    cv2.destroyAllWindows()

bgr_input = input('Inserisci intensità [0,255] per i 3 canali (Blue, Green, Red) separati da uno spazio: ')
bgr_vals = [int(val.strip()) for val in bgr_input.split(' ') ]

show_fullscreen_color(bgr_vals)