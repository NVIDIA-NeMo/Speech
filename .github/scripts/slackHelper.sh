#!/bin/bash

function sendSlackMessage() {

  WEBHOOK_URL="$1"
  PIPELINE_URL="$2"

  curl -X POST -H "Content-type: application/json" --data "{
      \"blocks\": [
        {
			\"type\": \"section\",
			\"text\": {
				\"type\": \"mrkdwn\",
				\"text\": \"\
🚨 *CI/CD failure at <$PIPELINE_URL|NeMo CI>*:

\"
			}
		}
      ]
    }" $WEBHOOK_URL

}
