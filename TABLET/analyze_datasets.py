#!/usr/bin/env python3
"""
Dataset Analysis Script for TABLET Implementation
Analyzes FinTabNet_OTSL and PubTabNet_OTSL datasets from HuggingFace
"""

import os
import json
from collections import Counter, defaultdict
from datasets import load_dataset
import numpy as np
from PIL import Image
import io
import base64

def analyze_dataset_info(dataset_name):
    """Get basic dataset information without downloading"""
    try:
        print(f"\n{'='*60}")
        print(f"ANALYZING {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Load dataset info only
        dataset_info = load_dataset(dataset_name, streaming=True)
        
        print("\n📋 DATASET SPLITS:")
        for split_name in dataset_info.keys():
            print(f"  - {split_name}")
        
        # Get first example to understand structure
        train_iter = iter(dataset_info['train'])
        first_example = next(train_iter)
        
        print("\n🔍 DATASET SCHEMA:")
        for key, value in first_example.items():
            value_type = type(value).__name__
            if isinstance(value, (list, dict)):
                if isinstance(value, list) and len(value) > 0:
                    inner_type = type(value[0]).__name__ if value else "empty"
                    print(f"  - {key}: {value_type}[{inner_type}] (length: {len(value)})")
                elif isinstance(value, dict):
                    print(f"  - {key}: {value_type} (keys: {list(value.keys())})")
                else:
                    print(f"  - {key}: {value_type}")
            else:
                print(f"  - {key}: {value_type}")
        
        return first_example, dataset_info
        
    except Exception as e:
        print(f"❌ Error analyzing {dataset_name}: {e}")
        return None, None

def analyze_sample_data(dataset_name, dataset_info, num_samples=100):
    """Analyze sample data to understand patterns"""
    print(f"\n📊 ANALYZING {num_samples} SAMPLES FROM {dataset_name.upper()}:")
    
    try:
        train_iter = iter(dataset_info['train'])
        
        # Collect statistics
        stats = {
            'otsl_lengths': [],
            'grid_sizes': [],
            'row_counts': [],
            'col_counts': [],
            'cell_counts': [],
            'otsl_tokens': Counter(),
            'image_sizes': [],
            'html_lengths': [],
            'table_complexities': []
        }
        
        for i, example in enumerate(train_iter):
            if i >= num_samples:
                break
                
            # OTSL analysis
            if 'otsl' in example and example['otsl']:
                otsl_sequence = example['otsl']
                stats['otsl_lengths'].append(len(otsl_sequence))
                stats['otsl_tokens'].update(otsl_sequence)
            
            # Grid analysis
            if 'row_num' in example and 'col_num' in example:
                rows = example['row_num']
                cols = example['col_num']
                stats['row_counts'].append(rows)
                stats['col_counts'].append(cols)
                stats['cell_counts'].append(rows * cols)
                stats['grid_sizes'].append(f"{rows}x{cols}")
                
                # Table complexity (simple heuristic)
                complexity = "simple" if rows * cols <= 50 else "complex"
                stats['table_complexities'].append(complexity)
            
            # Image analysis (if available)
            if 'image' in example and example['image']:
                try:
                    # Handle different image formats
                    if hasattr(example['image'], 'size'):
                        width, height = example['image'].size
                        stats['image_sizes'].append((width, height))
                except:
                    pass
            
            # HTML analysis
            if 'html' in example and example['html']:
                stats['html_lengths'].append(len(example['html']))
        
        # Print statistics
        print_statistics(stats, dataset_name)
        
        return stats
        
    except Exception as e:
        print(f"❌ Error analyzing samples: {e}")
        return None

def print_statistics(stats, dataset_name):
    """Print formatted statistics"""
    
    print(f"\n📈 STATISTICS FOR {dataset_name.upper()}:")
    
    # OTSL Statistics
    if stats['otsl_lengths']:
        otsl_lens = stats['otsl_lengths']
        print(f"\n🎯 OTSL Sequence Statistics:")
        print(f"  - Average length: {np.mean(otsl_lens):.1f}")
        print(f"  - Min/Max length: {min(otsl_lens)} / {max(otsl_lens)}")
        print(f"  - Median length: {np.median(otsl_lens):.1f}")
        print(f"  - Std deviation: {np.std(otsl_lens):.1f}")
    
    # Grid Statistics
    if stats['row_counts'] and stats['col_counts']:
        print(f"\n📏 Grid Size Statistics:")
        print(f"  - Average rows: {np.mean(stats['row_counts']):.1f}")
        print(f"  - Average cols: {np.mean(stats['col_counts']):.1f}")
        print(f"  - Average cells: {np.mean(stats['cell_counts']):.1f}")
        print(f"  - Max grid size: {max(stats['row_counts'])}x{max(stats['col_counts'])}")
        print(f"  - Min grid size: {min(stats['row_counts'])}x{min(stats['col_counts'])}")
        
        # Grid size distribution
        grid_counter = Counter(stats['grid_sizes'])
        print(f"  - Most common grid sizes:")
        for size, count in grid_counter.most_common(5):
            print(f"    • {size}: {count} tables")
    
    # Table complexity
    if stats['table_complexities']:
        complexity_counter = Counter(stats['table_complexities'])
        print(f"\n🔄 Table Complexity:")
        for complexity, count in complexity_counter.items():
            percentage = (count / len(stats['table_complexities'])) * 100
            print(f"  - {complexity}: {count} ({percentage:.1f}%)")
    
    # OTSL Token Distribution
    if stats['otsl_tokens']:
        print(f"\n🏷️  OTSL Token Distribution:")
        total_tokens = sum(stats['otsl_tokens'].values())
        for token, count in stats['otsl_tokens'].most_common():
            percentage = (count / total_tokens) * 100
            print(f"  - {token}: {count} ({percentage:.1f}%)")
    
    # Image sizes
    if stats['image_sizes']:
        widths = [size[0] for size in stats['image_sizes']]
        heights = [size[1] for size in stats['image_sizes']]
        print(f"\n🖼️  Image Size Statistics:")
        print(f"  - Average dimensions: {np.mean(widths):.0f}x{np.mean(heights):.0f}")
        print(f"  - Min dimensions: {min(widths)}x{min(heights)}")
        print(f"  - Max dimensions: {max(widths)}x{max(heights)}")
    
    # HTML lengths
    if stats['html_lengths']:
        print(f"\n📄 HTML Statistics:")
        print(f"  - Average HTML length: {np.mean(stats['html_lengths']):.0f} chars")
        print(f"  - Min/Max HTML length: {min(stats['html_lengths'])} / {max(stats['html_lengths'])}")

def analyze_otsl_patterns(dataset_name, dataset_info, num_samples=50):
    """Analyze OTSL token patterns and sequences"""
    print(f"\n🔍 ANALYZING OTSL PATTERNS FROM {dataset_name.upper()}:")
    
    try:
        train_iter = iter(dataset_info['train'])
        
        patterns = {
            'token_transitions': defaultdict(Counter),
            'sequence_patterns': Counter(),
            'span_patterns': Counter(),
            'empty_patterns': Counter()
        }
        
        for i, example in enumerate(train_iter):
            if i >= num_samples:
                break
                
            if 'otsl' in example and example['otsl']:
                otsl_sequence = example['otsl']
                
                # Token transitions
                for j in range(len(otsl_sequence) - 1):
                    current_token = otsl_sequence[j]
                    next_token = otsl_sequence[j + 1]
                    patterns['token_transitions'][current_token][next_token] += 1
                
                # Common patterns
                for j in range(len(otsl_sequence) - 2):
                    pattern = tuple(otsl_sequence[j:j+3])
                    patterns['sequence_patterns'][pattern] += 1
                
                # Spanning patterns
                span_tokens = ['lcel', 'ucel', 'xcel']
                for token in span_tokens:
                    if token in otsl_sequence:
                        patterns['span_patterns'][token] += 1
                
                # Empty cell patterns
                empty_count = otsl_sequence.count('ecel')
                patterns['empty_patterns'][empty_count] += 1
        
        # Print pattern analysis
        print(f"\n🔄 Token Transition Patterns (Top 5):")
        for token in ['fcel', 'ecel', 'lcel', 'ucel']:
            if token in patterns['token_transitions']:
                print(f"  After '{token}':")
                for next_token, count in patterns['token_transitions'][token].most_common(3):
                    print(f"    → {next_token}: {count}")
        
        print(f"\n📊 Most Common 3-Token Sequences:")
        for pattern, count in patterns['sequence_patterns'].most_common(10):
            print(f"  {' → '.join(pattern)}: {count}")
        
        print(f"\n🔗 Spanning Token Usage:")
        total_samples = sum(patterns['span_patterns'].values())
        for token, count in patterns['span_patterns'].items():
            percentage = (count / num_samples) * 100 if num_samples > 0 else 0
            print(f"  {token}: {count} tables ({percentage:.1f}%)")
        
        return patterns
        
    except Exception as e:
        print(f"❌ Error analyzing patterns: {e}")
        return None

def main():
    """Main analysis function"""
    print("🚀 TABLET Dataset Analysis Script")
    print("Analyzing ds4sd/FinTabNet_OTSL and ds4sd/PubTabNet_OTSL")
    
    datasets_to_analyze = [
        "ds4sd/FinTabNet_OTSL",
        "ds4sd/PubTabNet_OTSL"
    ]
    
    all_results = {}
    
    for dataset_name in datasets_to_analyze:
        try:
            # Basic dataset info
            first_example, dataset_info = analyze_dataset_info(dataset_name)
            
            if dataset_info is not None:
                # Sample data analysis
                stats = analyze_sample_data(dataset_name, dataset_info, num_samples=200)
                
                # Pattern analysis
                patterns = analyze_otsl_patterns(dataset_name, dataset_info, num_samples=100)
                
                all_results[dataset_name] = {
                    'first_example': first_example,
                    'stats': stats,
                    'patterns': patterns
                }
                
        except Exception as e:
            print(f"❌ Failed to analyze {dataset_name}: {e}")
            continue
    
    # Generate summary
    print(f"\n{'='*60}")
    print("📋 ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, results in all_results.items():
        if results['stats']:
            stats = results['stats']
            print(f"\n{dataset_name}:")
            if stats['otsl_lengths']:
                print(f"  - Average OTSL length: {np.mean(stats['otsl_lengths']):.1f}")
            if stats['cell_counts']:
                print(f"  - Average cells per table: {np.mean(stats['cell_counts']):.1f}")
            if stats['table_complexities']:
                complex_count = stats['table_complexities'].count('complex')
                total_count = len(stats['table_complexities'])
                print(f"  - Complex tables: {complex_count}/{total_count} ({(complex_count/total_count)*100:.1f}%)")
    
    print(f"\n✅ Analysis complete! Results ready for plan.md update.")
    
    # Save results to JSON for plan update
    with open('/home/ntlpt59/master/own/implementations/TABLET/dataset_analysis_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Counter):
                return dict(obj)
            elif isinstance(obj, defaultdict):
                return {str(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, dict):
                return {str(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_results = {}
        for dataset_name, results in all_results.items():
            serializable_results[dataset_name] = {
                'stats': convert_numpy(results['stats']) if results['stats'] else None,
                'patterns': convert_numpy(results['patterns']) if results['patterns'] else None
            }
        
        json.dump(serializable_results, f, indent=2)
    
    return all_results

if __name__ == "__main__":
    main()