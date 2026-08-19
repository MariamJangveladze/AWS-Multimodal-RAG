# Security baseline

This repository is a portfolio backend, but its AWS path follows production-minded defaults:

- AWS credentials are obtained from the standard SDK credential chain and are never stored in code.
- S3 objects remain private; image access uses short-lived pre-signed URLs.
- The runtime reads JSON metadata instead of deserializing untrusted pickle files.
- Query length and retrieval count are bounded to reduce accidental cost and abuse.
- API errors do not expose AWS exception details to callers.
- CORS is not enabled by default because this repository intentionally has no frontend.
- The Lambda or container role should receive only `s3:GetObject`, `s3:ListBucket`,
  `bedrock:InvokeModel`, and CloudWatch logging permissions for the named resources.

Before an internet-facing deployment, add authentication, rate limiting, WAF rules, dependency
scanning, CloudTrail alerting, data-classification controls, and a retention policy for request logs.

Do not log user prompts in regulated deployments until the data owner approves the logging policy.
