FROM fluent/fluentd:v1.16-debian-arm64-1

USER root

# Install Elasticsearch plugin
RUN gem install fluent-plugin-elasticsearch --no-document

USER fluent
