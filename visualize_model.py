#!/usr/bin/env python3
"""
Model Architecture Visualization
Creates visual representations of the emotion detection models
"""

import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def visualize_model_architecture(model_path, output_dir='visualizations'):
    """Create model architecture diagram"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    model_name = os.path.basename(model_path).split('.')[0]
    
    # Plot model architecture
    plot_model(
        model,
        to_file=f'{output_dir}/{model_name}_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',  # Top to Bottom
        expand_nested=True,
        dpi=200
    )
    print(f"Architecture diagram saved to: {output_dir}/{model_name}_architecture.png")
    
    # Create layer visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Layer types distribution
    layer_types = {}
    for layer in model.layers:
        layer_type = type(layer).__name__
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
    
    axes[0].bar(layer_types.keys(), layer_types.values())
    axes[0].set_title('Layer Types Distribution')
    axes[0].set_xlabel('Layer Type')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Parameters per layer (top 10)
    layer_params = []
    for layer in model.layers:
        if layer.count_params() > 0:
            layer_params.append((layer.name, layer.count_params()))
    
    layer_params.sort(key=lambda x: x[1], reverse=True)
    top_layers = layer_params[:10]
    
    if top_layers:
        names, params = zip(*top_layers)
        axes[1].barh(range(len(names)), params)
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels(names)
        axes[1].set_xlabel('Number of Parameters')
        axes[1].set_title('Top 10 Layers by Parameter Count')
        
        # Add value labels
        for i, v in enumerate(params):
            axes[1].text(v, i, f' {v:,}', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_layer_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Layer analysis saved to: {output_dir}/{model_name}_layer_analysis.png")
    
    # Print model summary
    print(f"\n=== Model Summary: {model_name} ===")
    model.summary()
    
    # Save model summary to text file
    with open(f'{output_dir}/{model_name}_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    return model

def visualize_feature_maps(model, sample_image, layer_names=None, output_dir='visualizations'):
    """Visualize feature maps from specific layers"""
    
    if layer_names is None:
        # Get convolutional layers
        layer_names = [layer.name for layer in model.layers 
                      if 'conv' in layer.name.lower()][:5]  # First 5 conv layers
    
    # Create feature extractor
    outputs = [model.get_layer(name).output for name in layer_names]
    feature_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    
    # Get features
    features = feature_model.predict(np.expand_dims(sample_image, axis=0))
    
    # Plot feature maps
    for layer_name, feature_map in zip(layer_names, features):
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        axes = axes.ravel()
        
        # Plot first 32 feature maps
        for i in range(min(32, feature_map.shape[-1])):
            axes[i].imshow(feature_map[0, :, :, i], cmap='viridis')
            axes[i].axis('off')
        
        plt.suptitle(f'Feature Maps: {layer_name}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/features_{layer_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"Feature maps saved to: {output_dir}/features_*.png")

def create_model_comparison(model_paths, output_dir='visualizations'):
    """Compare multiple models"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_data = []
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            model_name = os.path.basename(model_path).split('.')[0]
            
            # Collect metrics
            data = {
                'Model': model_name,
                'Parameters': model.count_params(),
                'Layers': len(model.layers),
                'Input Shape': str(model.input_shape),
                'Output Shape': str(model.output_shape),
                'Size (MB)': os.path.getsize(model_path) / (1024**2)
            }
            comparison_data.append(data)
    
    if comparison_data:
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        models = [d['Model'] for d in comparison_data]
        
        # Parameters comparison
        params = [d['Parameters'] for d in comparison_data]
        axes[0, 0].bar(models, params)
        axes[0, 0].set_title('Total Parameters')
        axes[0, 0].set_ylabel('Parameters')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Layers comparison
        layers = [d['Layers'] for d in comparison_data]
        axes[0, 1].bar(models, layers)
        axes[0, 1].set_title('Number of Layers')
        axes[0, 1].set_ylabel('Layers')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Size comparison
        sizes = [d['Size (MB)'] for d in comparison_data]
        axes[1, 0].bar(models, sizes)
        axes[1, 0].set_title('Model Size')
        axes[1, 0].set_ylabel('Size (MB)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Parameters per MB
        efficiency = [p/s for p, s in zip(params, sizes)]
        axes[1, 1].bar(models, efficiency)
        axes[1, 1].set_title('Parameters per MB')
        axes[1, 1].set_ylabel('Parameters/MB')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Model comparison saved to: {output_dir}/model_comparison.png")
        
        # Save comparison table
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
        print(f"Comparison table saved to: {output_dir}/model_comparison.csv")
        print("\n" + df.to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize model architecture and features')
    parser.add_argument('model_path', help='Path to the model file')
    parser.add_argument('--compare', nargs='+', help='Additional models to compare')
    parser.add_argument('--features', action='store_true', help='Visualize feature maps')
    parser.add_argument('--output', default='visualizations', help='Output directory')
    
    args = parser.parse_args()
    
    if os.path.exists(args.model_path):
        # Visualize main model
        model = visualize_model_architecture(args.model_path, args.output)
        
        # Compare models if specified
        if args.compare:
            all_models = [args.model_path] + args.compare
            create_model_comparison(all_models, args.output)
        
        # Visualize features if requested
        if args.features:
            # Create sample image
            sample_image = np.random.randn(48, 48, 3).astype(np.float32)
            visualize_feature_maps(model, sample_image, output_dir=args.output)
    else:
        print(f"Error: Model file not found: {args.model_path}")