"""
Mini-projet : Filtre FIR sur signal audio
- Lecture d'un fichier audio
- Application du filtre FIR
- Écoute et sauvegarde du résultat
- Visualisation temporelle et spectrale
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import sounddevice as sd
from scipy.io import wavfile
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# PARTIE 1 : CONCEPTION DU FILTRE FIR
# ============================================================

def concevoir_fir_passe_bas(Fe, Fc, M, fenetre_type='hamming'):
    """
    Conçoit un filtre FIR passe-bas.

    Paramètres:
        Fe : fréquence d'échantillonnage (Hz)
        Fc : fréquence de coupure (Hz)
        M  : ordre du filtre (longueur = M+1)
        fenetre_type : 'hamming', 'hann', 'blackman', 'rect'

    Retourne:
        h : coefficients du filtre
    """
    L = M + 1
    t = np.arange(L)
    fc_norm = Fc / (Fe / 2)

    # Réponse impulsionnelle idéale
    h_ideal = np.sinc(2 * fc_norm * (t - M / 2))

    # Fenêtre
    if fenetre_type == 'hamming':
        fenetre = np.hamming(L)
    elif fenetre_type == 'hann':
        fenetre = np.hanning(L)
    elif fenetre_type == 'blackman':
        fenetre = np.blackman(L)
    else:
        fenetre = np.ones(L)

    h = h_ideal * fenetre
    h = h / np.sum(h)  # normalisation

    return h


def concevoir_fir_passe_haut(Fe, Fc, M, fenetre_type='hamming'):
    """
    Conçoit un filtre FIR passe-haut.
    """
    # Passe-bas prototype
    h_pb = concevoir_fir_passe_bas(Fe, Fc, M, fenetre_type)

    # Transformation passe-bas -> passe-haut
    h_ph = h_pb * ((-1) ** np.arange(len(h_pb)))

    return h_ph


def concevoir_fir_passe_bande(Fe, Fc1, Fc2, M, fenetre_type='hamming'):
    """
    Conçoit un filtre FIR passe-bande.

    Paramètres:
        Fe : fréquence d'échantillonnage (Hz)
        Fc1 : fréquence de coupure basse (Hz)
        Fc2 : fréquence de coupure haute (Hz)
        M : ordre du filtre
    """
    L = M + 1
    t = np.arange(L)

    fc1_norm = Fc1 / (Fe / 2)
    fc2_norm = Fc2 / (Fe / 2)

    # Passe-bande idéal
    h_ideal = 2 * fc2_norm * np.sinc(2 * fc2_norm * (t - M / 2))
    h_ideal -= 2 * fc1_norm * np.sinc(2 * fc1_norm * (t - M / 2))

    # Fenêtre
    if fenetre_type == 'hamming':
        fenetre = np.hamming(L)
    elif fenetre_type == 'hann':
        fenetre = np.hanning(L)
    elif fenetre_type == 'blackman':
        fenetre = np.blackman(L)
    else:
        fenetre = np.ones(L)

    h = h_ideal * fenetre
    h = h / np.sum(np.abs(h))

    return h


# ============================================================
# PARTIE 2 : FILTRAGE AUDIO
# ============================================================

def appliquer_filtre_fir(audio, h):
    """
    Applique le filtre FIR à un signal audio.
    Utilise la convolution pour un filtrage propre.
    """
    # Convolution avec le filtre
    audio_filtre = np.convolve(audio, h, mode='same')

    # Normalisation pour éviter la saturation
    max_val = np.max(np.abs(audio_filtre))
    if max_val > 0:
        audio_filtre = audio_filtre / max_val * 0.95

    return audio_filtre


def filtrer_audio_par_blocs(audio, h, taille_bloc=1024):
    """
    Version par blocs pour les longs fichiers audio.
    Plus efficace en mémoire.
    """
    Lh = len(h)
    N = len(audio)
    y = np.zeros(N)

    # État initial (ligne à retard)
    etat = np.zeros(Lh)

    for i in range(0, N, taille_bloc):
        fin = min(i + taille_bloc, N)
        bloc = audio[i:fin]

        # Filtrage du bloc
        bloc_filtre = np.zeros(len(bloc))
        for n in range(len(bloc)):
            # Mise à jour de la ligne à retard
            etat = np.roll(etat, 1)
            etat[0] = bloc[n]

            # Produit scalaire
            bloc_filtre[n] = np.sum(h * etat)

        y[i:fin] = bloc_filtre

    # Normalisation
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val * 0.95

    return y


# ============================================================
# PARTIE 3 : VISUALISATION AUDIO
# ============================================================

def afficher_audio(audio_original, audio_filtre, Fe, titre="Filtrage audio"):
    """
    Affiche les formes d'onde et les spectres.
    """
    t = np.arange(len(audio_original)) / Fe

    plt.figure(figsize=(14, 10))

    # Formes d'onde
    plt.subplot(2, 2, 1)
    duree_affichee = min(0.1, len(t) / Fe)  # 100 ms
    echantillons = int(duree_affichee * Fe)
    plt.plot(t[:echantillons], audio_original[:echantillons])
    plt.title("Signal original (extrait)")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(t[:echantillons], audio_filtre[:echantillons], color='orange')
    plt.title("Signal filtré (extrait)")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Spectres
    N_fft = min(32768, len(audio_original))

    freq = np.fft.rfftfreq(N_fft, 1 / Fe)
    spec_orig = np.abs(np.fft.rfft(audio_original[:N_fft]))
    spec_filtre = np.abs(np.fft.rfft(audio_filtre[:N_fft]))

    plt.subplot(2, 2, 3)
    plt.semilogx(freq[1:], 20 * np.log10(spec_orig[1:] + 1e-10))
    plt.title("Spectre original")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.grid(True)
    plt.xlim(20, Fe / 2)

    plt.subplot(2, 2, 4)
    plt.semilogx(freq[1:], 20 * np.log10(spec_filtre[1:] + 1e-10), color='orange')
    plt.title("Spectre filtré")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.grid(True)
    plt.xlim(20, Fe / 2)

    plt.tight_layout()
    plt.show()


def afficher_reponse_filtre(h, Fe):
    """
    Affiche la réponse fréquentielle du filtre.
    """
    w, H = signal.freqz(h, worN=8000, fs=Fe)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(w, 20 * np.log10(abs(H) + 1e-10))
    plt.title("Réponse fréquentielle du filtre")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Gain (dB)")
    plt.grid(True)
    plt.ylim(-60, 5)

    plt.subplot(1, 2, 2)
    plt.plot(w, np.unwrap(np.angle(H)))
    plt.title("Réponse de phase")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Phase (rad)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# PARTIE 4 : LECTURE ET ÉCOUTE
# ============================================================

def lire_audio(chemin_fichier):
    """
    Lit un fichier audio.
    Supporte WAV et MP3 (nécessite pydub).
    """
    if chemin_fichier.endswith('.wav'):
        Fe, audio = wavfile.read(chemin_fichier)
        # Conversion en flottant entre -1 et 1
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        # Stéréo -> Mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        return Fe, audio

    elif chemin_fichier.endswith('.mp3'):
        try:
            from pydub import AudioSegment
            import io

            audio_seg = AudioSegment.from_mp3(chemin_fichier)
            audio = np.array(audio_seg.get_array_of_samples())
            Fe = audio_seg.frame_rate

            if audio_seg.channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)

            audio = audio.astype(np.float32) / (2 ** (8 * audio_seg.sample_width - 1))

            return Fe, audio
        except ImportError:
            print("❌ Installer pydub pour lire les MP3 : pip install pydub")
            return None, None

    else:
        print("Format non supporté. Utilisez WAV ou MP3.")
        return None, None


def sauvegarder_audio(chemin, Fe, audio):
    """
    Sauvegarde l'audio filtré en fichier WAV.
    """
    # Conversion en int16
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(chemin, Fe, audio_int16)
    print(f"✅ Audio sauvegardé : {chemin}")


def ecouter_audio(audio, Fe, duree_max=5):
    """
    Écoute l'audio (les premières secondes).
    """
    duree = min(len(audio) / Fe, duree_max)
    print(f"\n🔊 Lecture de {duree:.1f} secondes...")
    sd.play(audio[:int(duree * Fe)], Fe)
    sd.wait()


# ============================================================
# PARTIE 5 : INTERFACE PRINCIPALE
# ============================================================

def filtrer_audio_fichier(chemin_entree, chemin_sortie, type_filtre='pbas',
                          Fc=1000, Fc2=3000, M=101, fenetre='hamming'):
    """
    Fonction principale pour filtrer un fichier audio.

    Paramètres:
        chemin_entree : chemin du fichier audio source
        chemin_sortie : chemin pour sauvegarder le résultat
        type_filtre : 'pbas', 'phaut', 'pbande'
        Fc : fréquence de coupure (pour pbas/phaut) ou Fc1 (pour pbande)
        Fc2 : fréquence de coupure haute (pour pbande uniquement)
        M : ordre du filtre (plus M est grand, plus la coupure est nette)
        fenetre : 'hamming', 'hann', 'blackman'
    """
    print("\n" + "=" * 60)
    print("FILTRAGE AUDIO PAR FILTRE FIR")
    print("=" * 60)

    # 1. Lecture du fichier
    print(f"\n📁 Lecture du fichier : {chemin_entree}")
    Fe, audio = lire_audio(chemin_entree)

    if audio is None:
        return

    print(f"   Fréquence échantillonnage : {Fe} Hz")
    print(f"   Durée : {len(audio) / Fe:.2f} s")
    print(f"   Canaux : Mono")

    # 2. Conception du filtre
    print(f"\n🔧 Conception du filtre ({type_filtre})...")
    print(f"   Ordre M = {M} (longueur = {M + 1})")
    print(f"   Fenêtre : {fenetre}")

    if type_filtre == 'pbas':
        h = concevoir_fir_passe_bas(Fe, Fc, M, fenetre)
        print(f"   Fréquence de coupure : {Fc} Hz")
    elif type_filtre == 'phaut':
        h = concevoir_fir_passe_haut(Fe, Fc, M, fenetre)
        print(f"   Fréquence de coupure : {Fc} Hz")
    elif type_filtre == 'pbande':
        h = concevoir_fir_passe_bande(Fe, Fc, Fc2, M, fenetre)
        print(f"   Bande passante : [{Fc} - {Fc2}] Hz")
    else:
        print("Type de filtre inconnu")
        return

    # 3. Visualisation du filtre
    afficher_reponse_filtre(h, Fe)

    # 4. Application du filtre
    print("\n⚙️ Application du filtre...")
    audio_filtre = appliquer_filtre_fir(audio, h)

    # 5. Visualisation des résultats
    afficher_audio(audio, audio_filtre, Fe)

    # 6. Écoute
    print("\n🎧 Écoute du signal original...")
    ecouter_audio(audio, Fe, duree_max=3)

    print("\n🎧 Écoute du signal filtré...")
    ecouter_audio(audio_filtre, Fe, duree_max=3)

    # 7. Sauvegarde
    sauvegarder_audio(chemin_sortie, Fe, audio_filtre)

    print("\n" + "=" * 60)
    print("✅ Filtrage terminé !")
    print("=" * 60)

    return audio_filtre, Fe


# ============================================================
# PARTIE 6 : EXEMPLES D'UTILISATION
# ============================================================

def exemple_filtrage_passe_bas():
    """
    Exemple : filtre passe-bas pour enlever les aigus.
    """
    # À MODIFIER : mettre le chemin vers votre fichier audio
    fichier_entree = "mon_audio.wav"
    fichier_sortie = "mon_audio_filtre_pbas.wav"

    # Filtre passe-bas à 1000 Hz
    filtrer_audio_fichier(
        chemin_entree=fichier_entree,
        chemin_sortie=fichier_sortie,
        type_filtre='pbas',
        Fc=1000,  # Coupe à 1000 Hz
        M=101,  # Ordre (plus grand = coupure plus nette)
        fenetre='hamming'
    )


def exemple_filtrage_passe_haut():
    """
    Exemple : filtre passe-haut pour enlever les graves.
    """
    fichier_entree = "mon_audio.wav"
    fichier_sortie = "mon_audio_filtre_phaut.wav"

    # Filtre passe-haut à 300 Hz (enlève les bruits graves)
    filtrer_audio_fichier(
        chemin_entree=fichier_entree,
        chemin_sortie=fichier_sortie,
        type_filtre='phaut',
        Fc=300,
        M=101,
        fenetre='hamming'
    )


def exemple_filtrage_passe_bande():
    """
    Exemple : filtre passe-bande pour la voix (300 Hz - 3400 Hz).
    """
    fichier_entree = "mon_audio.wav"
    fichier_sortie = "mon_audio_filtre_pbande.wav"

    # Filtre passe-bande pour la voix humaine
    filtrer_audio_fichier(
        chemin_entree=fichier_entree,
        chemin_sortie=fichier_sortie,
        type_filtre='pbande',
        Fc=300,  # Fréquence coupure basse
        Fc2=3400,  # Fréquence coupure haute
        M=101,
        fenetre='hamming'
    )


def exemple_creation_signal_test():
    """
    Crée un fichier audio de test pour expérimenter.
    """
    Fe = 44100
    duree = 5
    t = np.linspace(0, duree, int(Fe * duree))

    # Signal composé de plusieurs fréquences
    signal = (0.5 * np.sin(2 * np.pi * 440 * t) +  # La3 (passe)
              0.3 * np.sin(2 * np.pi * 880 * t) +  # La4 (passe)
              0.2 * np.sin(2 * np.pi * 2000 * t) +  # Aigu (à couper)
              0.1 * np.sin(2 * np.pi * 5000 * t))  # Très aigu (à couper)

    # Normalisation
    signal = signal / np.max(np.abs(signal))

    # Sauvegarde
    sauvegarder_audio("signal_test.wav", Fe, signal)
    print("✅ Signal de test créé : signal_test.wav")

    return signal, Fe


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FILTRE FIR POUR AUDIO")
    print("=" * 60)

    # Créer un signal de test si vous n'avez pas de fichier audio
    print("\n🔊 Création d'un signal de test...")
    signal_test, Fe_test = exemple_creation_signal_test()

    # Appliquer un filtre passe-bas au signal de test
    print("\n📊 Test du filtre passe-bas sur signal synthétique...")
    h = concevoir_fir_passe_bas(Fe_test, Fc=1000, M=101)
    signal_filtre = appliquer_filtre_fir(signal_test, h)

    # Visualisation
    afficher_audio(signal_test, signal_filtre, Fe_test)

    # Écoute
    print("\n🎧 Écoute du signal original (mélange de fréquences)...")
    ecouter_audio(signal_test, Fe_test, duree_max=3)

    print("\n🎧 Écoute du signal filtré (seulement les basses)...")
    ecouter_audio(signal_filtre, Fe_test, duree_max=3)

    # Sauvegarde
    sauvegarder_audio("signal_filtre_pbas.wav", Fe_test, signal_filtre)

    # ============================================================
    # DÉCOMMENTEZ POUR UTILISER VOTRE PROPRE FICHIER AUDIO
    # ============================================================

    # Remplacer "mon_audio.wav" par le chemin de votre fichier
    # fichier_audio = "votre_fichier.wav"
    #
    # filtrer_audio_fichier(
    #     chemin_entree=fichier_audio,
    #     chemin_sortie="audio_filtre.wav",
    #     type_filtre='pbas',  # 'pbas', 'phaut', ou 'pbande'
    #     Fc=1000,
    #     Fc2=3000,
    #     M=101,
    #     fenetre='hamming'
    # )