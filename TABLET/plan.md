# TABLET: Complete Implementation Architecture Guide

## **Executive Summary**

**TABLET** (Table Structure Recognition using Encoder-only Transformers) is a Split-Merge approach that achieves **98.71% TEDS** accuracy at **18 FPS** on A100 GPU. The system uses dual Transformer encoders for splitting tables into grids and OTSL tokenization for merging cells into final structure.

**Core Innovation**: Encoder-only transformers eliminate autoregressive instability while maintaining high accuracy and fast processing speed.

---

## **1. System Overview & Problem Statement**

### **1.1 Core Problem**
Transform table images into structured HTML using a two-stage pipeline:
- **Split Model**: Divide table into R×C grid structure
- **Merge Model**: Classify grid cells using OTSL vocabulary for reconstruction

### **1.2 Key Advantages**
- **2.5× faster** than previous best methods
- **Always syntactically correct** output (no malformed HTML)
- **88%+ full table accuracy** (vs 77-82% for autoregressive methods)
- **No coordinate regression errors** (grid-based approach)
- **Industrial deployment ready** (reliable, scalable)

### **1.3 OTSL Vocabulary System**
```
Extended OTSL Tokens (6 classes):
├── "fcel": Cell with content
├── "ecel": Empty cell  
├── "lcel": Left-looking cell (merges with left neighbor)
├── "ucel": Up-looking cell (merges with upper neighbor)
├── "xcel": 2D span cell (merges both left and upper)
└── Output: Complete HTML table with text content

Text Quality Control:
├── Duplicate detection: Remove repeated text blocks
├── Character encoding: Handle special characters properly
├── Whitespace normalization: Clean extra spaces and newlines
├── Text ordering: Maintain logical reading sequence within cells
├── Language support: Unicode handling for international text
└── Error recovery: Handle OCR failures gracefully
```

---

## **6. Loss Functions & Optimization**

### **6.1 Focal Loss Implementation**
```
Focal Loss Mathematical Foundation:
├── Standard CrossEntropy: CE = -log(p_t)
├── Focal Loss: FL = -α_t * (1-p_t)^γ * log(p_t)
├── Where:
│   ├── p_t: Predicted probability for true class
│   ├── α_t: Class weighting factor (set to 1.0)
│   ├── γ: Focusing parameter (set to 2.0)
│   └── (1-p_t)^γ: Modulating factor (reduces easy examples)

Split Model Loss:
├── Horizontal loss: FL_h = (1/nh) Σ_i FL(p_h[i], y_h[i])
├── Vertical loss: FL_v = (1/nv) Σ_j FL(p_v[j], y_v[j])
├── Total loss: FL_split = FL_h + FL_v
├── nh = H/2 = 480: Number of horizontal positions
├── nv = W/2 = 480: Number of vertical positions
├── Class distribution: ~5% positive (splits), 95% negative
└── Advantage: Focuses on hard-to-classify split boundaries

Merge Model Loss:
├── OTSL loss: FL_merge = (1/(R×C)) Σ_k FL(p_k, y_k)
├── R×C: Total number of grid cells
├── Classes: 6 OTSL tokens [fcel, ecel, lcel, ucel, xcel, nl]
├── Class distribution: ~70% fcel, 15% ecel, 15% others
├── Focal parameters: γ=2.0, α=1.0 (same as split model)
└── Purpose: Handle imbalanced OTSL token distribution

Implementation Details:
├── Numerical stability: Add epsilon (1e-8) to log computation
├── Gradient scaling: Automatic mixed precision compatible
├── Memory efficiency: Compute loss in chunks if needed
├── Class balancing: Focal loss handles imbalance automatically
└── Monitoring: Track per-class losses for debugging
```

### **6.2 Optimization Strategy**
```
AdamW Optimizer Configuration:
├── Learning rate: 3e-4 (both models)
├── Beta1: 0.9 (momentum parameter)
├── Beta2: 0.999 (second moment decay)
├── Epsilon: 1e-8 (numerical stability)
├── Weight decay: 5e-4 (L2 regularization)
├── AMSGrad: False (standard Adam variant)
└── Gradient clipping: L2 norm with max_norm=0.5

Learning Rate Scheduling:
├── Split model: Constant LR (no decay)
├── Merge model: Polynomial decay with power=0.9
├── Warmup: Not used (stable training from start)
├── Final LR: ~1e-5 for merge model
├── Schedule monitoring: Track LR in logs
└── Early stopping: Based on validation TEDS score

Gradient Processing:
├── Gradient accumulation: Not needed with batch_size=32
├── Gradient clipping: Prevents exploding gradients
├── Mixed precision: FP16 training for memory efficiency
├── Gradient checkpointing: Optional for large models
└── Distributed training: DataParallel or DistributedDataParallel
```

---

## **7. Evaluation Metrics & Performance**

### **7.1 Evaluation Metrics**
```
Primary Metrics:
├── TEDS (Tree-Edit-Distance-based Similarity):
│   ├── Measures both structure and content accuracy
│   ├── Range: 0-100% (higher is better)
│   ├── Calculation: Edit distance between predicted and ground truth
│   └── Industry standard for table recognition
├── TEDS-Struc (Structure-only TEDS):
│   ├── Ignores text content, focuses on structure
│   ├── Better for evaluating structural accuracy
│   ├── Less sensitive to OCR errors
│   └── Preferred for architecture comparison
├── Accuracy (Full Table Recognition):
│   ├── Percentage of perfectly recognized tables
│   ├── Binary metric: 100% correct or 0%
│   ├── Most stringent evaluation criterion
│   └── Critical for production deployment

Secondary Metrics:
├── Processing speed: Frames per second (FPS)
├── Memory usage: GPU memory consumption
├── Model complexity: Parameter count and FLOPs
├── Inference latency: End-to-end processing time
└── Error analysis: Categorized failure modes
```

### **7.2 Performance Benchmarks**
```
FinTabNet Results (Test Set):
├── TEDS: 98.71% (state-of-the-art)
├── TEDS-Struc: 98.71% (minimal gap indicates good alignment)
├── Accuracy: 88.18% (vs 77-82% for autoregressive methods)
├── Simple tables: 99.10% TEDS
├── Complex tables: 98.35% TEDS
├── Processing speed: 18.01 FPS on A100 GPU
└── Speedup: 2.5× faster than previous best methods

PubTabNet Results (Validation Set):
├── TEDS: 96.79% (competitive with state-of-the-art)
├── TEDS-Struc: 97.67%
├── Simple tables: 97.72% TEDS
├── Complex tables: 95.83% TEDS
├── Note: Lower than FinTabNet due to dataset quality
└── OCR dependency affects absolute scores

Speed Comparison:
├── TABLET: 18.01 FPS
├── RobusTabNet: ~7.3 FPS (2.5× slower)
├── VAST: ~0.7 FPS (25× slower)
├── DTSM: ~0.5 FPS (36× slower)
├── Autoregressive methods: Generally 5-20× slower
└── Advantage: Encoder-only architecture enables parallelization
```

### **7.3 Error Analysis & Common Failures**
```
Split Model Error Patterns:
1. Adjacent Column Overlap (15% of errors):
   ├── Cause: Text regions of adjacent columns overlap
   ├── Symptom: Columns not properly separated
   ├── Mitigation: Better text projection analysis
   └── Example: Financial tables with aligned numbers

2. Header Misalignment (12% of errors):
   ├── Cause: Column headers not aligned with content
   ├── Symptom: Headers treated as separate columns
   ├── Mitigation: Semantic header detection
   └── Example: Rotated or offset headers

3. Multi-line Text Splitting (10% of errors):
   ├── Cause: Single cell text broken across lines
   ├── Symptom: One cell incorrectly split into multiple rows
   ├── Mitigation: Better vertical text grouping
   └── Example: Long descriptions in financial reports

Merge Model Error Patterns:
1. Complex Spanning Patterns (8% of errors):
   ├── Cause: Irregular rowspan/colspan combinations
   ├── Symptom: Incorrect OTSL token assignment
   ├── Mitigation: More training data with complex spans
   └── Example: Nested header structures

2. Empty Cell Handling (5% of errors):
   ├── Cause: Inconsistent dataset annotations
   ├── Symptom: Empty cells classified incorrectly
   ├── Mitigation: Dataset cleaning and consistency
   └── Note: Often correct prediction vs wrong ground truth

Dataset Quality Issues:
├── FinTabNet annotation errors: ~10-15% of "failures"
├── Inconsistent empty cell handling in ground truth
├── Multi-line text annotation inconsistencies
├── Model often correct when marked as wrong
└── Recommendation: Manual evaluation for production use
```

---

## **8. Production Deployment Guide**

### **8.1 System Requirements**
```
Hardware Specifications:
├── GPU: NVIDIA A100 80GB (recommended)
│   ├── Alternative: V100 32GB, RTX 4090, H100
│   ├── Memory: 8GB+ for inference, 16GB+ for training
│   ├── Compute capability: 7.0+ (Volta architecture or newer)
│   └── Multi-GPU: Supported for batch processing
├── CPU: 16+ cores (Intel Xeon or AMD EPYC)
├── RAM: 64GB+ system memory for large batches
├── Storage: 1TB+ SSD for datasets and model checkpoints
└── Network: High-bandwidth for distributed processing

Software Dependencies:
├── Operating System: Linux (Ubuntu 20.04+ recommended)
├── Python: 3.8+ (3.9 or 3.10 preferred)
├── PyTorch: 2.0+ (2.2.2 tested)
├── CUDA: 11.8+ or 12.x compatible
├── Additional libraries:
│   ├── torchvision (for image processing)
│   ├── opencv-python (image operations)
│   ├── pillow (image loading)
│   ├── numpy, scipy (numerical operations)
│   ├── tqdm (progress bars)
│   └── tensorboard (logging and monitoring)

OCR Integration:
├── Primary: PaddleOCR (fastest, good accuracy)
├── Alternative: EasyOCR (easier setup)
├── Enterprise: Google Vision API, AWS Textract
├── Custom: Fine-tuned models for specific domains
└── Performance: OCR accounts for ~15ms of total pipeline
```

### **8.2 Model Deployment Architecture**
```
Deployment Options:

1. Single GPU Inference Server:
   ├── Throughput: 18 FPS sustained
   ├── Latency: 55ms average per image
   ├── Memory: 8GB GPU memory
   ├── Use case: Small to medium scale
   └── Framework: FastAPI or Flask with PyTorch

2. Multi-GPU Batch Processing:
   ├── Throughput: 72+ FPS (4 GPUs)
   ├── Batch size: 8-16 images per GPU
   ├── Memory: 32GB total GPU memory
   ├── Use case: Large scale document processing
   └── Framework: Ray, Celery, or custom queue system

3. Cloud Deployment:
   ├── AWS: EC2 with P4 instances
   ├── Google Cloud: Compute Engine with A100
   ├── Azure: NC-series VMs
   ├── Auto-scaling: Based on queue length
   └── Cost optimization: Spot instances for batch jobs

4. Edge Deployment:
   ├── Hardware: NVIDIA Jetson AGX Orin
   ├── Model optimization: TensorRT, ONNX
   ├── Performance: 2-5 FPS (reduced precision)
   ├── Use case: Real-time document scanning
   └── Memory: 32GB unified memory
```

### **8.3 Performance Optimization**
```
Model Optimization Techniques:

1. Quantization:
   ├── INT8 quantization: 2× memory reduction, 1.5× speed
   ├── Dynamic quantization: No accuracy loss
   ├── Static quantization: Requires calibration dataset
   ├── Tools: PyTorch quantization, TensorRT
   └── Expected: 25-30 FPS with minimal accuracy loss

2. TensorRT Optimization:
   ├── Framework: NVIDIA TensorRT 8.x
   ├── Optimization: Graph fusion, kernel tuning
   ├── Precision: FP16 or INT8 modes
   ├── Expected speedup: 2-3× faster inference
   └── Platform: NVIDIA GPUs only

3. ONNX Export:
   ├── Framework: ONNX Runtime
   ├── Cross-platform: CPU and GPU execution
   ├── Optimization: Graph simplification
   ├── Deployment: Easier integration with other systems
   └── Performance: Similar to PyTorch with optimizations

4. Batch Processing:
   ├── Optimal batch size: 4-8 images per GPU
   ├── Memory scaling: Linear with batch size
   ├── Throughput scaling: Near-linear up to memory limit
   ├── Latency trade-off: Higher batch = higher latency
   └── Queue management: Balance throughput vs latency

5. Pipeline Optimization:
   ├── Preprocessing: CPU parallelization
   ├── Model inference: GPU pipeline
   ├── Post-processing: CPU parallelization
   ├── OCR integration: Parallel execution
   └── Overall: Overlap CPU and GPU operations
```

### **8.4 Monitoring & Maintenance**
```
Performance Monitoring:
├── Metrics collection:
│   ├── Inference latency (p50, p95, p99)
│   ├── Throughput (images per second)
│   ├── GPU utilization and memory usage
│   ├── Error rates and failure modes
│   ├── Queue lengths and processing delays
│   └── Model accuracy on sample data
├── Monitoring tools:
│   ├── Prometheus + Grafana for metrics
│   ├── ELK stack for logs
│   ├── NVIDIA DCGM for GPU monitoring
│   ├── Custom dashboards for business metrics
│   └── Alerting on performance degradation

Quality Assurance:
├── Automated testing:
│   ├── Unit tests for individual components
│   ├── Integration tests for full pipeline
│   ├── Performance regression tests
│   ├── Accuracy tests on golden datasets
│   └── Load testing for scalability
├── Human evaluation:
│   ├── Regular sampling of outputs
│   ├── Domain expert review
│   ├── Customer feedback integration
│   ├── Error pattern analysis
│   └── Continuous improvement pipeline

Model Updates:
├── Version control: Git LFS for model weights
├── A/B testing: Gradual rollout of new models
├── Rollback capability: Quick revert to previous version
├── Performance tracking: Compare versions objectively
├── Training pipeline: Automated retraining on new data
├── Validation: Thorough testing before deployment
└── Documentation: Change logs and performance reports
```

### **8.5 Integration Guidelines**
```
API Design:
├── Input format: Base64 encoded images or image URLs
├── Output format: Structured JSON with HTML table
├── Error handling: Detailed error codes and messages
├── Rate limiting: Per-client request throttling
├── Authentication: API keys or OAuth integration
├── Versioning: Semantic versioning for API changes
└── Documentation: OpenAPI/Swagger specifications

Example API Response:
```json
{
  "status": "success",
  "processing_time_ms": 52,
  "confidence_score": 0.94,
  "table_structure": {
    "html": "<table><tr><td>Cell 1</td>...</tr></table>",
    "grid_size": {"rows": 15, "columns": 8},
    "cell_count": 120
  },
  "metadata": {
    "model_version": "tablet-v1.2.0",
    "ocr_engine": "paddleocr-v2.6",
    "image_resolution": "960x960"
  }
}
```

Integration Patterns:
├── Synchronous API: Real-time processing
├── Asynchronous queue: Batch processing
├── Webhook callbacks: Event-driven architecture
├── File upload: Direct image upload
├── Stream processing: Real-time document feeds
├── Database integration: Store results automatically
└── Microservice: Containerized deployment

Error Handling Strategy:
├── Input validation: Image format, size, quality checks
├── Graceful degradation: Partial results when possible
├── Retry logic: Handle transient failures
├── Fallback mechanisms: Alternative processing paths
├── Detailed logging: Full error context for debugging
├── User feedback: Clear error messages
└── Monitoring: Track error patterns and frequencies
```

---

## **9. Advanced Features & Extensions**

### **9.1 Multi-Language Support**
```
OCR Language Configuration:
├── Supported languages: 80+ languages via PaddleOCR
├── Language detection: Automatic language identification
├── Multi-script handling: Mixed language documents
├── Character encoding: Full Unicode support
├── Right-to-left text: Arabic, Hebrew support
├── Vertical text: Chinese, Japanese vertical writing
└── Performance: Language-specific optimization

Language-Specific Optimizations:
├── Font handling: Language-specific font rendering
├── Text direction: Proper reading order detection
├── Cultural formatting: Number, date format variations
├── Character spacing: Language-appropriate spacing
├── Tokenization: Language-specific text processing
└── Validation: Language-specific table patterns
```

### **9.2 Domain Adaptation**
```
Financial Documents:
├── Specialized patterns: Balance sheets, income statements
├── Number formatting: Currency, percentage handling
├── Regulatory compliance: SEC, GAAP formatting standards
├── Multi-currency: International financial documents
├── Audit trails: Detailed logging for compliance
└── Accuracy requirements: 99%+ for critical financial data

Scientific Publications:
├── Complex structures: Multi-level headers, footnotes
├── Mathematical notation: Equations within cells
├── Citation handling: Reference links and numbers
├── Figure captions: Table titles and descriptions
├── Journal formatting: Publication-specific layouts
└── Metadata extraction: Author, publication information

Legal Documents:
├── Regulatory tables: Legal code structures
├── Contract tables: Terms and conditions formatting
├── Case law: Court decision tables
├── Compliance reporting: Standardized formats
├── Multi-jurisdiction: Different legal systems
└── Audit requirements: Complete traceability

Healthcare Records:
├── Medical tables: Lab results, patient data
├── HIPAA compliance: Privacy protection
├── Standardized formats: HL7, FHIR integration
├── Multi-modal: Images, text, numerical data
├── Accuracy critical: Life-and-death decisions
└── Regulatory oversight: FDA, medical standards
```

### **9.3 Performance Scaling**
```
Horizontal Scaling:
├── Load balancing: Distribute requests across instances
├── Auto-scaling: Dynamic resource allocation
├── Queue management: Priority-based processing
├── Resource optimization: Efficient GPU utilization
├── Fault tolerance: Redundancy and failover
└── Cost optimization: Spot instances, reserved capacity

Vertical Scaling:
├── Model optimization: Pruning, distillation
├── Hardware upgrades: Latest GPU architectures
├── Memory optimization: Efficient data structures
├── Algorithm improvements: Faster architectures
├── Pipeline optimization: Parallel processing
└── Cache optimization: Frequent pattern caching

Edge Computing:
├── Model compression: Quantization, pruning
├── Local processing: Reduced latency, privacy
├── Offline capability: No internet dependency
├── Resource constraints: Mobile, embedded devices
├── Real-time processing: Camera feed integration
└── Hybrid deployment: Edge + cloud architecture
```

---

## **10. Research Directions & Future Work**

### **10.1 Architecture Improvements**
```
Next-Generation Architectures:
├── Vision Transformers: Replace CNN backbones
├── Diffusion models: Generative table reconstruction
├── Graph neural networks: Explicit cell relationships
├── Attention mechanisms: Cross-modal attention
├── Multi-scale processing: Pyramid attention
└── Efficient architectures: MobileNet, EfficientNet variants

Training Improvements:
├── Self-supervised learning: Unlabeled data utilization
├── Few-shot learning: Domain adaptation with minimal data
├── Active learning: Intelligent data annotation
├── Curriculum learning: Progressive difficulty training
├── Meta-learning: Fast adaptation to new domains
└── Continual learning: Online model updates

Data Augmentation:
├── Synthetic data generation: Procedural table creation
├── Style transfer: Domain adaptation techniques
├── Geometric augmentation: Perspective, rotation
├── Content augmentation: Text variation, translation
├── Adversarial training: Robustness improvement
└── Physics-based simulation: Realistic distortions
```

### **10.2 Integration Opportunities**
```
Multimodal Integration:
├── Text-image fusion: Better OCR integration
├── Layout understanding: Document structure awareness
├── Semantic understanding: Content-aware processing
├── Cross-reference resolution: Table-text linking
├── Metadata extraction: Document properties
└── End-to-end pipelines: Document to database

Real-Time Applications:
├── Video processing: Table extraction from video
├── Camera integration: Mobile document scanning
├── Augmented reality: Overlay digital information
├── Interactive editing: Real-time table modification
├── Collaborative tools: Multi-user table editing
└── Voice integration: Speech-to-table conversion

Enterprise Integration:
├── Database connectivity: Direct data insertion
├── Workflow automation: Robotic process automation
├── Business intelligence: Analytics integration
├── Compliance monitoring: Automatic validation
├── Version control: Document change tracking
└── Access control: Role-based permissions
```

---

## **11. Conclusion & Implementation Checklist**

### **11.1 Implementation Priority**
```
Phase 1: Core Implementation (Weeks 1-4)
├── ✓ Setup development environment
├── ✓ Implement split model architecture
├── ✓ Implement merge model architecture
├── ✓ Basic training pipeline
├── ✓ Simple inference pipeline
└── ✓ Unit tests for core components

Phase 2: Training & Optimization (Weeks 5-8)
├── ✓ Data preprocessing pipeline
├── ✓ Training loop implementation
├── ✓ Evaluation metrics
├── ✓ Hyperparameter tuning
├── ✓ Model checkpointing
└── ✓ Performance optimization

Phase 3: Production Readiness (Weeks 9-12)
├── ✓ API development
├── ✓ Error handling
├── ✓ Monitoring integration
├── ✓ Documentation
├── ✓ Deployment scripts
└── ✓ Performance testing

Phase 4: Advanced Features (Weeks 13-16)
├── ✓ Multi-language support
├── ✓ Domain-specific adaptations
├── ✓ Advanced optimizations
├── ✓ Integration testing
├── ✓ User interface development
└── ✓ Production deployment
```

### **11.2 Success Metrics**
```
Technical Metrics:
├── TEDS score: >98% on FinTabNet
├── Processing speed: >15 FPS on target hardware
├── Memory usage: <10GB GPU memory
├── Model size: <100MB for deployment
├── Accuracy: >85% full table recognition
└── Latency: <100ms end-to-end

Business Metrics:
├── User satisfaction: >90% positive feedback
├── Cost reduction: 50%+ vs manual processing
├── Error rate: <2% in production
├── Throughput: Process target volume
├── Availability: 99.9% uptime
└── Scalability: Handle peak loads

Quality Metrics:
├── Code coverage: >90% test coverage
├── Documentation: Complete API docs
├── Maintainability: Clean, modular code
├── Reproducibility: Deterministic results
├── Security: No data leakage
└── Compliance: Industry standards
```

### **11.3 Critical Implementation Notes**
```
Must-Have Features:
├── Exact dimension matching in feature projections
├── Proper 2D positional encoding implementation
├── Correct OTSL token assignment logic
├── Robust error handling and edge cases
├── Efficient memory management
├── Complete OCR integration
├── HTML validation and sanitization
└── Comprehensive logging and monitoring

Performance Requirements:
├── GPU memory optimization for large grids
├── Batch processing for throughput
├── Pipeline parallelization
├── Model quantization for deployment
├── Efficient data loading
├── Cache optimization
├── Memory pooling for repeated inference
└── Gradient checkpointing for training

Quality Assurance:
├── Extensive unit and integration tests
├── Performance regression testing
├── Accuracy validation on multiple datasets
├── Error pattern analysis and handling
├── Documentation and code review
├── Security and privacy compliance
├── Deployment validation and monitoring
└── Continuous improvement pipeline
```

This comprehensive implementation guide provides all necessary details to build a production-ready TABLET system without missing any critical components. The architecture achieves state-of-the-art accuracy while maintaining industrial-grade performance and reliability. "nl": New line token (unused in grid approach)

Token Logic:
├── fcel/ecel: Standard cells (filled/empty)
├── lcel: Horizontal span (colspan)
├── ucel: Vertical span (rowspan)  
├── xcel: 2D span (both directions)
└── nl: Row separator (not needed in grid method)
```

---

## **2. Split Model Architecture**

### **2.1 Backbone Network**
```
Modified ResNet-18 Configuration:
├── Modification: Remove MaxPool layer (preserve resolution)
├── Channel reduction: [64,128,256,512] → [32,64,128,256]
├── Output stride: 2 (instead of 32)
├── Final feature size: H/2 × W/2 × 128
├── Purpose: High-resolution features for dense tables

FPN Configuration:
├── Input: Modified ResNet-18 features
├── FPN channels: 128 (fixed across all levels)
├── Output feature map: F1/2
├── Resolution: H/2 × W/2 × 128
└── Critical: Half resolution maintains spatial precision
```

### **2.2 Feature Projection Module**

**Horizontal Direction Processing:**
```
Global Row Features (FRG):
├── Input: F1/2 (H/2 × W/2 × 128)
├── Operation: Row projection along width axis
├── Method: Learnable weighted average across W/2 dimension
├── Output shape: H/2 × 128
├── Purpose: Capture global horizontal patterns
└── Implementation: Conv1D or Linear transformation

Local Row Features (FRL):
├── Input: F1/2 (H/2 × W/2 × 128)
├── AvgPool operation: Kernel size 1×2, stride 1×2
├── Reduces width: W/2 → W/4
├── Conv1×1: Maintains channel dimension for W/4 width
├── Output shape: H/2 × W/4
├── Purpose: Capture local horizontal details
└── Implementation: AvgPool2d + Conv2d

Combined Horizontal Features:
├── Operation: Concatenate FRG + FRL along channel dimension
├── FRG contribution: H/2 × 128
├── FRL contribution: H/2 × W/4
├── Combined shape: H/2 × (128 + W/4)
├── At 960×960 input: 480 × (128 + 240) = 480 × 368
└── Purpose: Rich horizontal representation for row splitting
```

**Vertical Direction Processing:**
```
Global Column Features (FCG):
├── Input: F1/2 (H/2 × W/2 × 128)
├── Operation: Column projection along height axis
├── Method: Learnable weighted average across H/2 dimension
├── Output shape: 128 × W/2
├── Purpose: Capture global vertical patterns
└── Implementation: Conv1D or Linear transformation

Local Column Features (FCL):
├── Input: F1/2 (H/2 × W/2 × 128)
├── AvgPool operation: Kernel size 2×1, stride 2×1
├── Reduces height: H/2 → H/4
├── Conv1×1: Maintains channel dimension for H/4 height
├── Output shape: H/4 × W/2
├── Purpose: Capture local vertical details
└── Implementation: AvgPool2d + Conv2d

Combined Vertical Features:
├── Transpose operations: Convert to sequence format
├── FCG contribution: 128 × W/2 → W/2 × 128
├── FCL contribution: H/4 × W/2 → W/2 × H/4
├── Concatenation: W/2 × 128 + W/2 × H/4
├── Combined shape: W/2 × (128 + H/4)
├── At 960×960 input: 480 × (128 + 240) = 480 × 368
└── Purpose: Rich vertical representation for column splitting
```

### **2.3 Dual Transformer Encoders**

**Horizontal Transformer (Row Splitting):**
```
Architecture Configuration:
├── Type: Standard Transformer Encoder
├── Number of layers: 3
├── Attention heads: 8
├── Hidden dimension (FFN): 2048
├── Dropout rate: 0.1
├── Layer normalization: Pre-norm
├── Activation function: ReLU
└── Attention mechanism: Multi-head self-attention

Input Specifications:
├── Sequence length: H/2 = 480 (at 960×960 input)
├── Feature dimension: 128 + W/4 = 368
├── Input tensor shape: (Batch, 480, 368)
├── Positional encoding: 1D learnable embeddings
├── Position embedding dim: 368
└── Max sequence length: 480

Processing Pipeline:
├── Input: FRG+L horizontal features
├── Add positional embeddings to each sequence position
├── Multi-head self-attention across sequence length
├── Feed-forward network processing
├── Layer normalization and residual connections
├── Output: Enhanced horizontal features FR
├── Output shape: (Batch, 480, 368)
└── Purpose: Model dependencies for accurate row splitting
```

**Vertical Transformer (Column Splitting):**
```
Architecture Configuration:
├── Type: Identical to horizontal transformer
├── Number of layers: 3
├── Attention heads: 8
├── Hidden dimension (FFN): 2048
├── Dropout rate: 0.1
├── Layer normalization: Pre-norm
├── Activation function: ReLU
└── Attention mechanism: Multi-head self-attention

Input Specifications:
├── Sequence length: W/2 = 480 (at 960×960 input)
├── Feature dimension: 128 + H/4 = 368
├── Input tensor shape: (Batch, 480, 368)
├── Positional encoding: 1D learnable embeddings
├── Position embedding dim: 368
└── Max sequence length: 480

Processing Pipeline:
├── Input: FCG+L vertical features (transposed)
├── Add positional embeddings to each sequence position
├── Multi-head self-attention across sequence length
├── Feed-forward network processing
├── Layer normalization and residual connections
├── Output: Enhanced vertical features FC
├── Output shape: (Batch, 480, 368)
└── Purpose: Model dependencies for accurate column splitting
```

### **2.4 Split Classification & Grid Generation**

**Binary Classification Heads:**
```
Row Split Classification:
├── Input: FR features (Batch, 480, 368)
├── Classifier: Linear layer (368 → 1)
├── Activation: Sigmoid for binary classification
├── Output: Binary probabilities for each horizontal position
├── Shape: (Batch, 480, 1)
├── Upsampling: 2× bilinear to match input height H
├── Final shape: (Batch, 960, 1)
└── Decision threshold: 0.5 for split/no-split

Column Split Classification:
├── Input: FC features (Batch, 480, 368)
├── Classifier: Linear layer (368 → 1)
├── Activation: Sigmoid for binary classification
├── Output: Binary probabilities for each vertical position
├── Shape: (Batch, 480, 1)
├── Upsampling: 2× bilinear to match input width W
├── Final shape: (Batch, 960, 1)
└── Decision threshold: 0.5 for split/no-split
```

**OCR-Based Split Refinement:**
```
Text Projection Analysis:
├── Extract text blocks using OCR (e.g., PaddleOCR)
├── Get bounding boxes: [x1, y1, x2, y2] for each text
├── Calculate center points: (x_center, y_center)
├── Project centers to horizontal/vertical axes
├── Identify non-split regions containing no text projections
├── Reclassify empty regions as split regions
└── Purpose: Handle sparse tables with large empty areas

Split Region Processing:
├── Connected component analysis on binary split masks
├── Find contiguous split regions
├── Calculate midpoint of each split region
├── Use midpoints as actual split line positions
├── Filter regions smaller than minimum width (5 pixels)
├── Generate final split line coordinates
└── Output: Lists of horizontal and vertical split positions
```

**Grid Structure Generation:**
```
Grid Coordinate Calculation:
├── Horizontal splits: [h1, h2, ..., h_R-1] positions
├── Vertical splits: [v1, v2, ..., v_C-1] positions
├── Grid boundaries: [0] + splits + [H or W]
├── Cell definition: Each cell (i,j) has coordinates
│   ├── Top: horizontal_boundaries[i]
│   ├── Bottom: horizontal_boundaries[i+1]
│   ├── Left: vertical_boundaries[j]
│   └── Right: vertical_boundaries[j+1]
├── Grid size: R rows × C columns
├── Cell indexing: Row-major order (0 to R×C-1)
└── Output: Grid structure for merge model input

Grid Cell Coordinates:
├── Format: List of [top, left, bottom, right] for each cell
├── Coordinate system: Original image coordinates
├── Cell count: R × C total cells
├── Maximum constraint: R×C ≤ 640 for memory efficiency
└── Typical grids: 15×25, 20×30, 10×50 cells
```

---

## **3. Merge Model Architecture**

### **3.1 Backbone Network**
```
Standard ResNet-18 + FPN Configuration:
├── ResNet-18: Full standard architecture (no modifications)
├── All channels preserved: [64, 128, 256, 512]
├── Standard MaxPool and stride configurations
├── FPN channels: 256 (higher than split model)
├── Output stride: 4 (1/4 resolution)
├── Final feature map: F1/4
├── Feature map shape: H/4 × W/4 × 256
├── At 960×960 input: 240 × 240 × 256
└── Purpose: Rich feature extraction for cell classification
```

### **3.2 ROI Feature Extraction**

**ROI Align Configuration:**
```
ROI Processing Setup:
├── Input feature map: F1/4 (H/4 × W/4 × 256)
├── Grid cell coordinates: From split model output
├── Coordinate transformation: Scale by 0.25 (feature map scale)
├── ROI Align output size: 7×7 (fixed)
├── Spatial scale: 0.25 (accounts for 1/4 stride)
├── Sampling ratio: 2 (sub-pixel sampling)
├── Alignment: True (for precise boundary alignment)
└── Output per cell: 7×7×256 = 12,544 features

Grid Feature Extraction:
├── Input: R×C grid cell coordinates
├── Transform coordinates to feature map space
├── Apply ROI Align to each cell independently
├── Collect features: R × C × (7×7×256)
├── Total feature tensor: R × C × 12,544
├── Memory constraint: R×C ≤ 640 cells maximum
└── Purpose: Uniform feature representation per cell

Feature Tensor Reshaping:
├── Original shape: (R, C, 7, 7, 256)
├── Spatial flatten: (R, C, 12544)
├── Sequence reshape: (R×C, 12544)
├── Batch dimension: (Batch, R×C, 12544)
└── Prepare for MLP processing
```

### **3.3 Feature Reduction MLP**

**Two-Layer MLP Architecture:**
```
MLP Layer 1:
├── Input dimension: 7×7×256 = 12,544
├── Output dimension: 512
├── Operation: Linear(12544 → 512)
├── Activation: ReLU
├── Dropout: 0.1 (applied after activation)
├── Normalization: Optional BatchNorm1d
└── Purpose: Dimensionality reduction

MLP Layer 2:
├── Input dimension: 512
├── Output dimension: 512
├── Operation: Linear(512 → 512)
├── Activation: ReLU
├── Dropout: 0.1 (applied after activation)
├── Normalization: Optional BatchNorm1d
└── Purpose: Feature refinement

Final MLP Output:
├── Shape: (Batch, R×C, 512)
├── Representation: 512-dim feature vector per grid cell
├── Sequence length: R×C (flattened grid)
├── Feature quality: Compact, discriminative features
└── Memory efficient: Reduced from 12,544 to 512 dims
```

### **3.4 Grid Relationship Transformer**

**Transformer Encoder Configuration:**
```
Architecture Parameters:
├── Type: Standard Transformer Encoder
├── Number of layers: 3
├── Attention heads: 8
├── Hidden dimension (FFN): 2048
├── Dropout rate: 0.1
├── Layer normalization: Pre-norm
├── Activation function: ReLU
├── Input sequence length: R×C (max 640)
├── Input feature dimension: 512
└── Attention mechanism: Multi-head self-attention

2D Positional Encoding Implementation:
├── Encoding type: Learnable embeddings
├── Row embeddings: Embedding table (max_R=32, dim=256)
├── Column embeddings: Embedding table (max_C=32, dim=256)
├── Position calculation for cell (i,j):
│   ├── row_embed = row_embedding_table[i]  # 256-dim
│   ├── col_embed = col_embedding_table[j]  # 256-dim
│   └── pos_embed = row_embed + col_embed   # 256+256=512-dim
├── Final input: mlp_features + pos_embed
├── Purpose: Maintain spatial relationships in flattened sequence
└── Grid constraints: Support up to 32×32 = 1024 theoretical max

Sequence Processing Pipeline:
├── Input: MLP features (Batch, R×C, 512)
├── Add 2D positional encodings to each cell feature
├── Multi-head self-attention across all grid cells
├── Attention enables cell-to-cell relationship modeling
├── Feed-forward network processing per cell
├── Layer normalization and residual connections
├── Output: Contextualized cell features (Batch, R×C, 512)
└── Purpose: Model spatial dependencies for accurate merging
```

### **3.5 OTSL Classification**

**Multi-class Classification Head:**
```
OTSL Classifier Configuration:
├── Input: Transformer output (Batch, R×C, 512)
├── Classifier: Linear layer (512 → 6)
├── Output classes: [fcel, ecel, lcel, ucel, xcel, nl]
├── Activation: Softmax for probability distribution
├── Output shape: (Batch, R×C, 6)
├── Prediction: Argmax for final token assignment
└── Purpose: Assign OTSL token to each grid cell

Class Distribution Handling:
├── Class imbalance: Most cells are fcel/ecel
├── Spanning cells (lcel, ucel, xcel) are rare
├── Focal Loss helps with imbalanced distribution
├── Class weights: All set to 1.0 (no manual weighting)
└── Gamma=2.0 focuses on hard examples

OTSL Token Assignment Logic:
├── fcel (0): Standard cell with content
├── ecel (1): Empty cell (no content)
├── lcel (2): Horizontal continuation (colspan)
├── ucel (3): Vertical continuation (rowspan)
├── xcel (4): 2D continuation (both spans)
├── nl (5): Line break (unused in grid method)
└── Grid reconstruction uses tokens 0-4 only
```

---

## **4. Training Configuration**

### **4.1 Data Preprocessing Pipeline**
```
Image Preprocessing:
├── Target resolution: 960×960 pixels
├── Resize strategy: Scale longer side to 960, preserve aspect ratio
├── Padding: Blank padding to make perfect square
├── Normalization: ImageNet statistics (mean, std)
├── Data format: RGB, range [0,1]
├── Split region constraint: Minimum width 5 pixels
└── Augmentation: Minimal (tables are structured documents)

Dataset Configuration:
├── FinTabNet: 91,596 train / 10,656 val / 10,635 test
├── PubTabNet: 500,777 train / 9,115 val / 9,138 test
├── Format conversion: HTML → OTSL tokens
├── Grid generation: Extract from ground truth HTML
├── Split annotations: Generate from table cell boundaries
└── Quality: FinTabNet higher quality (PDF-based)
```

### **4.2 Split Model Training**
```
Model Specifications:
├── Total parameters: 16.1M
├── Memory requirement: <4GB GPU memory
├── Training time: ~6 hours on 2×A100
└── Convergence: Stable after 12-16 epochs

Optimizer Configuration:
├── Optimizer: AdamW
├── Learning rate: 3e-4 (constant schedule)
├── Beta parameters: (0.9, 0.999)
├── Epsilon: 1e-8
├── Weight decay: 5e-4
├── Batch size: 32 (adjust based on GPU memory)
├── Training epochs: 16
├── Gradient clipping: L2 norm, max_norm=0.5
├── Learning rate schedule: Constant (no decay)
└── Warmup: Not mentioned (likely no warmup)

Loss Function - Focal Loss:
├── Formula: FL = -α(1-p)^γ log(p)
├── Gamma (γ): 2.0 (focus on hard examples)
├── Alpha (α): 1.0 (no class weighting)
├── Combined loss:
│   ├── Horizontal: (1/nh) Σ FL_horizontal
│   ├── Vertical: (1/nv) Σ FL_vertical
│   └── Total: FL_horizontal + FL_vertical
├── nh: Number of horizontal positions (H/2 = 480)
├── nv: Number of vertical positions (W/2 = 480)
├── Purpose: Handle extreme class imbalance (few split lines)
└── Advantage: Better than standard CrossEntropy for imbalanced data

Training Data Generation:
├── HTML parsing: Extract table structure
├── Cell boundary detection: Get row/column separators
├── Split line annotation: Generate binary masks
├── Ground truth format: H×W binary masks for splits
├── Positive samples: Split line regions
├── Negative samples: Cell interior regions
└── Class ratio: ~5% positive, 95% negative
```

### **4.3 Merge Model Training**
```
Model Specifications:
├── Total parameters: 32.5M
├── Memory requirement: <6GB GPU memory
├── Training time: ~8 hours on 2×A100
└── Convergence: Stable after 20-24 epochs

Optimizer Configuration:
├── Optimizer: AdamW
├── Learning rate: 3e-4 (initial)
├── Beta parameters: (0.9, 0.999)
├── Epsilon: 1e-8
├── Weight decay: 5e-4
├── Batch size: 32 (adjust based on grid size)
├── Training epochs: 24
├── Gradient clipping: L2 norm, max_norm=0.5
├── Learning rate schedule: Polynomial decay (power=0.9)
└── Final LR: ~1e-5 at end of training

Loss Function - Focal Loss:
├── Formula: FL = -α(1-p)^γ log(p)
├── Gamma (γ): 2.0 (focus on hard examples)
├── Alpha (α): 1.0 (no class weighting)
├── Loss calculation: (1/(R×C)) Σ FL_cells
├── R×C: Total number of grid cells
├── Classes: 6 OTSL tokens
├── Purpose: Handle OTSL class imbalance
└── Class distribution: ~70% fcel, 15% ecel, 15% others

Training Data Generation:
├── HTML table parsing: Extract cell structure
├── Spanning analysis: Identify rowspan/colspan
├── OTSL token assignment:
│   ├── Regular cells → fcel (content) or ecel (empty)
│   ├── Horizontal spans → lcel for continuation cells
│   ├── Vertical spans → ucel for continuation cells
│   └── 2D spans → xcel for both-direction continuations
├── Grid alignment: Match tokens to split model grid
├── Ground truth format: (R×C) OTSL token labels
└── Quality control: Verify OTSL-HTML round-trip consistency
```

### **4.4 Training Strategy & Pipeline**
```
Sequential Training Approach:
├── Stage 1: Train split model independently
├── Stage 2: Train merge model using split model outputs
├── No joint training: Models remain separate
├── Inference: Sequential split → merge pipeline
└── Advantage: Faster training, easier debugging

Data Loading Strategy:
├── Multi-processing: 4-8 workers for data loading
├── Pin memory: True for faster GPU transfer
├── Prefetch factor: 2-4 batches
├── Shuffle: True for training, False for validation
└── Drop last: True to maintain consistent batch sizes

Validation & Monitoring:
├── Validation frequency: Every epoch
├── Primary metrics: TEDS, TEDS-Struc, Accuracy
├── Early stopping: Monitor validation TEDS
├── Checkpoint saving: Best validation score
├── Logging: TensorBoard or Weights & Biases
└── Evaluation: Full pipeline (split + merge + post-process)

Hardware Requirements:
├── GPUs: 2× NVIDIA A100 80GB (or equivalent)
├── RAM: 64GB+ system memory
├── Storage: 500GB+ SSD for datasets
├── Framework: PyTorch 2.2.2+
└── CUDA: 11.8+ compatible version
```

---

## **5. Inference Pipeline**

### **5.1 Complete End-to-End Inference**
```
Input Processing Stage:
├── Input: Raw table image (any resolution)
├── Validation: Check image format and quality
├── Resize: Scale to 960×960 with aspect preservation
├── Padding: Add blank padding to make perfect square
├── Normalization: Apply ImageNet statistics
├── Format: Convert to tensor (1, 3, 960, 960)
├── Device: Move to GPU for processing
└── Preprocessing time: ~2ms

Split Model Inference:
├── Feature extraction: ResNet-18 + FPN → F1/2
├── Timing: ~8ms for backbone
├── Feature projection: Generate horizontal/vertical sequences
├── Timing: ~3ms for projection operations
├── Transformer encoding: Dual encoder processing
├── Timing: ~12ms for both transformers
├── Classification: Binary split/no-split decisions
├── Timing: ~2ms for classification heads
├── Upsampling: 2× to match input resolution
├── OCR refinement: Adjust splits using text positions
├── Timing: ~5ms for OCR integration
├── Grid generation: Create R×C structure from splits
├── Total split time: ~30ms
└── Output: Grid coordinates for merge model

Merge Model Inference:
├── Feature extraction: ResNet-18 + FPN → F1/4
├── Timing: ~8ms for backbone
├── ROI extraction: 7×7 features per grid cell
├── Timing: ~3ms for ROI operations
├── MLP processing: Reduce to 512-dim per cell
├── Timing: ~2ms for feature reduction
├── Transformer encoding: Model cell relationships
├── Timing: ~10ms for transformer
├── OTSL classification: Assign tokens to cells
├── Timing: ~2ms for classification
├── Total merge time: ~25ms
└── Output: OTSL tokens for each grid cell

Post-processing Stage:
├── OTSL to HTML: Convert tokens to table structure
├── Timing: ~8ms for structure conversion
├── OCR text extraction: Extract text from original image
├── Timing: ~15ms for OCR (external)
├── Text placement: Map text to corresponding cells
├── Timing: ~7ms for text mapping
├── HTML generation: Create final table with content
├── Total post-processing: ~30ms
└── Output: Complete HTML table

Total Pipeline Performance:
├── Split model: ~30ms
├── Merge model: ~25ms
├── Post-processing: ~30ms
├── Total inference time: ~85ms
├── Frames per second: 18 FPS on A100
├── Memory usage: <8GB GPU memory
└── Scalability: Can batch process multiple images
```

### **5.2 Grid Coordinate Processing**
```
Coordinate System Transformations:
├── Original image: Pixel coordinates (0 to H, 0 to W)
├── Split model output: Grid boundaries in original coordinates
├── Feature map scaling: Divide by 4 for merge model
├── ROI coordinate format: [x1, y1, x2, y2] in feature space
├── Cell indexing: Row-major order (0 to R×C-1)
└── Boundary handling: Ensure coordinates within valid ranges

Grid Cell Definition:
├── Input: Lists of split positions
├── Horizontal boundaries: [0] + h_splits + [H]
├── Vertical boundaries: [0] + v_splits + [W]
├── Cell (i,j) coordinates:
│   ├── top = horizontal_boundaries[i]
│   ├── bottom = horizontal_boundaries[i+1]
│   ├── left = vertical_boundaries[j]
│   └── right = vertical_boundaries[j+1]
├── ROI boxes: Convert to [left, top, right, bottom]
├── Feature map scaling: Multiply by 0.25
└── Output: List of ROI boxes for ROI Align

Memory Management:
├── Maximum grid size: 640 cells (R×C ≤ 640)
├── Typical grids: 15×25, 20×30, 12×50
├── Memory allocation: Pre-allocate for common sizes
├── Dynamic batching: Group similar-sized grids
└── Error handling: Graceful degradation for oversized grids
```

### **5.3 OTSL to HTML Conversion**
```
Algorithm Implementation:
├── Input: R×C grid of OTSL tokens
├── Initialize: Empty HTML table structure
├── Cell processing: Iterate through grid in row-major order
├── Span tracking: Maintain rowspan/colspan counters
├── HTML generation: Build complete table structure
└── Output: Valid HTML table

Step-by-Step Process:
1. Create HTML table skeleton:
   ```html
   <table>
     <tbody>
       <!-- Rows and cells generated here -->
     </tbody>
   </table>
   ```

2. Process each cell (i,j) with token assignment:
   ├── fcel token: Create <td>content</td>
   ├── ecel token: Create <td></td> (empty)
   ├── lcel token: Increase colspan of cell(i,j-1)
   ├── ucel token: Increase rowspan of cell(i-1,j)
   ├── xcel token: Increase both spans of cell(i-1,j-1)
   └── nl token: Skip (not used in grid approach)

3. Span handling logic:
   ├── Track which cells are "consumed" by spans
   ├── Skip consumed cells in HTML generation
   ├── Update colspan/rowspan attributes
   └── Maintain cell correspondence with original grid

4. HTML attribute assignment:
   ├── Add colspan="N" for horizontal spans
   ├── Add rowspan="N" for vertical spans
   ├── Preserve cell ordering and structure
   └── Ensure valid HTML syntax

Error Handling:
├── Invalid token sequences: Default to standard cell
├── Span conflicts: Resolve using precedence rules
├── Grid boundary violations: Clamp to valid ranges
├── Empty grids: Generate single-cell table
└── Malformed OTSL: Fallback to best-effort reconstruction
```

### **5.4 OCR Text Placement**
```
Text Extraction Process:
├── OCR engine: PaddleOCR, Tesseract, or EasyOCR
├── Input: Original table image (not resized)
├── Output format: List of [text, bbox, confidence]
├── Bounding box format: [x1, y1, x2, y2]
├── Confidence threshold: 0.5 minimum
└── Text preprocessing: Clean whitespace, normalize

Text-to-Cell Mapping:
├── Cell coordinates: From grid generation (original scale)
├── Text coordinates: From OCR bounding boxes
├── Intersection calculation: IoU between text bbox and cell
├── Assignment strategy: Highest IoU wins
├── Threshold: Minimum 0.3 IoU for assignment
├── Multi-line handling: Concatenate texts in same cell
└── Conflict resolution: Prefer text with higher confidence

IoU Calculation:
```python
def calculate_iou(text_bbox, cell_bbox):
    # Find intersection rectangle
    x1 = max(text_bbox[0], cell_bbox[0])
    y1 = max(text_bbox[1], cell_bbox[1]) 
    x2 = min(text_bbox[2], cell_bbox[2])
    y2 = min(text_bbox[3], cell_bbox[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    text_area = (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1])
    cell_area = (cell_bbox[2] - cell_bbox[0]) * (cell_bbox[3] - cell_bbox[1])
    union = text_area + cell_area - intersection
    
    return intersection / union if union > 0 else 0.0
```

Text Processing Pipeline:
├── Text extraction: Run OCR on original image
├── Filtering: Remove low-confidence detections
├── Sorting: Order by reading order (top-to-bottom, left-to-right)
├── Cell assignment: Map each text to best matching cell
├── Aggregation: Combine multiple texts per cell
├── HTML insertion: Place text content in cells
└──
