"""
Estrae 50 frame casuali (ma distribuiti) da ogni video in data/personal
per preparare dataset da annotare
"""
import cv2
import numpy as np
from pathlib import Path
import random

def extract_frames_from_video(video_path, output_dir, num_frames=50):
    """
    Estrae num_frames frame distribuiti uniformemente dal video
    
    Args:
        video_path: Path al video
        output_dir: Directory dove salvare i frame
        num_frames: Numero di frame da estrarre
    """
    print(f"\n{'='*60}")
    print(f"📹 Video: {video_path.name}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Impossibile aprire {video_path.name}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"  Frame totali: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Durata: {duration:.1f}s")
    
    # Calcola frame da estrarre (distribuiti uniformemente)
    # Evita primi e ultimi 5% per evitare transizioni/titoli
    start_frame = int(total_frames * 0.05)
    end_frame = int(total_frames * 0.95)
    
    # Distribuisci uniformemente
    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    
    # Aggiungi un po' di randomness (±5 frame) per variabilità
    frame_indices = [max(start_frame, min(end_frame, idx + random.randint(-5, 5))) 
                     for idx in frame_indices]
    
    # Rimuovi duplicati e ordina
    frame_indices = sorted(list(set(frame_indices)))
    
    print(f"  Estraendo {len(frame_indices)} frame distribuiti...")
    
    extracted = 0
    video_stem = video_path.stem.replace(' ', '_').replace('(', '').replace(')', '')
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Nome file descrittivo
            output_filename = f"{video_stem}_frame{frame_idx:06d}.jpg"
            output_path = output_dir / output_filename
            
            cv2.imwrite(str(output_path), frame)
            extracted += 1
            
            if extracted % 10 == 0:
                print(f"  Estratti: {extracted}/{len(frame_indices)}")
    
    cap.release()
    
    print(f"  ✅ Completato: {extracted} frame salvati")
    return extracted

def main():
    # Directory video e output
    video_dir = Path("data/personal")
    output_dir = video_dir / "frames_for_annotation"
    
    # Crea directory output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("ESTRAZIONE FRAME PER ANNOTAZIONE")
    print("="*70)
    print(f"\nDirectory video: {video_dir}")
    print(f"Directory output: {output_dir}")
    print(f"Frame per video: 50 (distribuiti uniformemente)")
    
    # Trova tutti i video
    video_extensions = ['*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.AVI', '*.mkv', '*.MKV']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(ext))
    
    # Escludi sottodirectory
    video_files = [v for v in video_files if v.parent == video_dir]
    
    print(f"\n📹 Video trovati: {len(video_files)}")
    for video in sorted(video_files):
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"  - {video.name} ({size_mb:.0f} MB)")
    
    if not video_files:
        print("\n❌ Nessun video trovato in data/personal!")
        return
    
    # Chiedi conferma
    print(f"\n⚠️  Verranno estratti ~{len(video_files) * 50} frame totali")
    print(f"   Spazio disco stimato: ~{len(video_files) * 50 * 0.5:.0f} MB")
    
    # Processa tutti i video
    total_extracted = 0
    for video_path in sorted(video_files):
        extracted = extract_frames_from_video(video_path, output_dir, num_frames=50)
        total_extracted += extracted
    
    # Summary
    print("\n" + "="*70)
    print("RIEPILOGO")
    print("="*70)
    print(f"\n✅ Estrazione completata!")
    print(f"   Video processati: {len(video_files)}")
    print(f"   Frame estratti: {total_extracted}")
    print(f"   Salvati in: {output_dir}")
    
    print(f"\n📝 PROSSIMI PASSI:")
    print(f"   1. Vai su https://roboflow.com")
    print(f"   2. Crea nuovo progetto 'Padel Field Keypoints - Personal'")
    print(f"   3. Upload i frame da: {output_dir}")
    print(f"   4. Annota i keypoints seguendo lo schema:")
    print(f"      - Class 0: Barrier_keypoint (barriere)")
    print(f"      - Class 1: Field_keypoint (linee campo)")
    print(f"      - Class 2: Net_keypoint (rete)")
    print(f"      - Class 3: Wall_keypoint (muri)")
    print(f"   5. Esporta dataset in formato YOLO v8")
    print(f"   6. Scarica e metti in data/personal_keypoints/")
    
    # Crea file README con istruzioni
    readme_content = f"""# Frame per Annotazione - Padel Field Keypoints

## Frame Estratti
- Video processati: {len(video_files)}
- Frame totali: {total_extracted}
- Frame per video: ~50 (distribuiti uniformemente)

## Video Sorgente
{chr(10).join([f'- {v.name}' for v in sorted(video_files)])}

## Annotazione su Roboflow

### 1. Setup Progetto
1. Vai su https://roboflow.com
2. Crea nuovo progetto
   - Nome: "Padel Field Keypoints - Personal"
   - Project Type: "Object Detection"
   - Annotation Type: "Keypoint Detection"

### 2. Configurazione Keypoints
Definisci 4 classi con 1 keypoint ciascuna:
- **Barrier_keypoint** (Class 0): Punti delle barriere laterali
- **Field_keypoint** (Class 1): Punti delle linee del campo
- **Net_keypoint** (Class 2): Punti della rete centrale
- **Wall_keypoint** (Class 3): Punti dei muri di vetro/muratura

### 3. Upload Frame
- Upload tutti i {total_extracted} frame da questa cartella
- Roboflow li organizzerà automaticamente

### 4. Annotazione
Per ogni frame:
1. Identifica tutti i keypoints visibili del campo
2. Per ogni keypoint:
   - Disegna bounding box intorno alla regione
   - Marca il keypoint esatto al centro
   - Assegna la classe corretta

Suggerimenti:
- Annota solo keypoints chiaramente visibili
- Sii consistente nella nomenclatura
- ~2-3 ore per completare tutti i frame

### 5. Export Dataset
1. Generate → Export Dataset
2. Format: YOLO v8
3. Preprocessing: Nessuno (già fatto)
4. Augmentation: Opzionale (consigliato: flip, rotate ±15°)
5. Download ZIP

### 6. Fine-tuning
Dopo download:
1. Estrai ZIP in `data/personal_keypoints/`
2. Esegui script di fine-tuning (verrà creato)
"""
    
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_content)
    print(f"\n📄 README con istruzioni salvato: {readme_path}")

if __name__ == "__main__":
    main()
