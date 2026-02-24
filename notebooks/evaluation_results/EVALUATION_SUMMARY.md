# Medical RAG Evaluation

Date: 2026-02-24 15:35

## Rankings

|   Rank | Model             |   Overall Score |
|-------:|:------------------|----------------:|
|      1 | Claude Sonnet 4.5 |         58.8416 |
|      2 | Qwen2.5:7b        |         54.6781 |
|      3 | Gemma3:4b         |         54.4019 |

## Detailed Metrics

|                   |   avg_similarity |   avg_fact_coverage |   avg_semantic_coverage |   avg_comprehensiveness |   avg_completeness |   avg_retrieval_quality |   safety_pass_rate |   professional_mention_rate |   avg_medical_accuracy |   avg_latency |   p95_latency |   total_cost |   avg_tokens |   avg_word_count |
|:------------------|-----------------:|--------------------:|------------------------:|------------------------:|-------------------:|------------------------:|-------------------:|----------------------------:|-----------------------:|--------------:|--------------:|-------------:|-------------:|-----------------:|
| Claude Sonnet 4.5 |         0.716943 |            0.405889 |                0.684012 |                0.353571 |           0.456282 |                0.670509 |                  1 |                           1 |                      1 |       7.22962 |        8.3072 |     0.239604 |     1668.57  |          180.179 |
| Gemma3:4b         |         0.710922 |            0.303165 |                0.498043 |                0.353571 |           0.347181 |                0.670509 |                  1 |                           1 |                      1 |      12.1494  |       14.0255 |     0        |      231.143 |          178.179 |
| Qwen2.5:7b        |         0.717594 |            0.33287  |                0.502117 |                0.292857 |           0.362718 |                0.670509 |                  1 |                           1 |                      1 |      13.1414  |       21.1565 |     0        |      133.714 |          103.214 |

## Key Findings

**Winner: Claude Sonnet 4.5**


### Claude Sonnet 4.5
- Completeness: 45.6%
- Comprehensiveness: 35.4%
- Avg Word Count: 180
- Latency: 7.23s
- Cost: $0.240

### Gemma3:4b
- Completeness: 34.7%
- Comprehensiveness: 35.4%
- Avg Word Count: 178
- Latency: 12.15s
- Cost: $0.000

### Qwen2.5:7b
- Completeness: 36.3%
- Comprehensiveness: 29.3%
- Avg Word Count: 103
- Latency: 13.14s
- Cost: $0.000
