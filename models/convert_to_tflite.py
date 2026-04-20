import os
import re
from pathlib import Path
import tensorflow as tf

def get_model_directories():
    """Get available model directories."""
    models_dir = Path(__file__).parent
    dirs = [d for d in ['Combined', 'MyoGym', 'RecoFit'] 
            if (models_dir / d).is_dir()]
    return dirs

def get_models_in_directory(directory):
    """Get all .keras files in a directory."""
    dir_path = Path(__file__).parent / directory
    models = sorted([f.name for f in dir_path.glob('*.keras')])
    return models

def extract_timestamps_from_filename(filename):
    """Extract timestamps from filename for preservation."""
    # Remove .keras extension
    name_without_ext = filename.replace('.keras', '')
    return name_without_ext

def convert_model(model_path, output_path):
    """Convert Keras model to TFLite format."""
    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    print(f"Saving converted model to: {output_path}")
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    print("✓ Conversion complete!")

def main():
    # Get available directories
    directories = get_model_directories()
    
    if not directories:
        print("Error: No model directories found (Combined, MyoGym, RecoFit)")
        return
    
    print("\nAvailable model directories:")
    for i, dir_name in enumerate(directories, 1):
        print(f"  {i}. {dir_name}")
    
    # User selects directory
    while True:
        try:
            dir_choice = int(input("\nSelect directory (number): ")) - 1
            if 0 <= dir_choice < len(directories):
                selected_dir = directories[dir_choice]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Get models in selected directory
    models = get_models_in_directory(selected_dir)
    
    if not models:
        print(f"Error: No .keras models found in {selected_dir}/")
        return
    
    print(f"\nAvailable models in {selected_dir}/:")
    for i, model_name in enumerate(models, 1):
        print(f"  {i}. {model_name}")
    
    # User selects model
    while True:
        try:
            model_choice = int(input("\nSelect model (number): ")) - 1
            if 0 <= model_choice < len(models):
                selected_model = models[model_choice]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Prepare paths
    model_dir = Path(__file__).parent / selected_dir
    model_path = model_dir / selected_model
    
    # Create output filename with same timestamps
    base_name = extract_timestamps_from_filename(selected_model)
    output_filename = f"{base_name}.tflite"
    output_path = model_dir / output_filename
    
    # Check if output already exists
    if output_path.exists():
        response = input(f"\n{output_filename} already exists. Overwrite? (y/n): ").lower()
        if response != 'y':
            print("Conversion cancelled.")
            return
    
    # Convert and save
    convert_model(str(model_path), str(output_path))

if __name__ == "__main__":
    main()
