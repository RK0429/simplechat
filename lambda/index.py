# lambda/index.py
import json
import os
import re  # 正規表現モジュールをインポート
import urllib.error
import urllib.request


# Lambda コンテキストからリージョンを抽出する関数
def extract_region_from_arn(arn):
    # ARN 形式: arn:aws:lambda:region:account-id:function:function-name
    match = re.search("arn:aws:lambda:([^:]+):", arn)
    if match:
        return match.group(1)
    return "us-east-1"  # デフォルト値


# External FastAPI endpoint URL from environment variable
FASTAPI_URL = os.environ.get("FASTAPI_URL")


# Helper function to call the external FastAPI inference API
def call_external_api(message: str) -> dict:
    if not FASTAPI_URL:
        raise ValueError("FASTAPI_URL environment variable is not set")
    payload = {"message": message}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        FASTAPI_URL, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp_body = resp.read().decode("utf-8")
            return json.loads(resp_body)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP error from external API: {e.code} {e.reason}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error when calling external API: {e.reason}")


def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))

        # Cognitoで認証されたユーザー情報を取得
        user_info = None
        if "requestContext" in event and "authorizer" in event["requestContext"]:
            user_info = event["requestContext"]["authorizer"]["claims"]
            print(
                f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}"
            )

        # リクエストボディの解析
        body = json.loads(event["body"])
        message = body["message"]
        conversation_history = body.get("conversationHistory", [])

        print("Processing message:", message)

        # 会話履歴を使用
        messages = conversation_history.copy()

        # ユーザーメッセージを追加
        messages.append({"role": "user", "content": message})

        # Call external FastAPI for inference
        api_result = call_external_api(message)
        assistant_response = api_result.get("response")
        if assistant_response is None:
            raise Exception("Invalid API response: 'response' field is missing")

        # アシスタントの応答を会話履歴に追加
        messages.append({"role": "assistant", "content": assistant_response})

        # 成功レスポンスの返却
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST",
            },
            "body": json.dumps(
                {
                    "success": True,
                    "response": assistant_response,
                    "conversationHistory": messages,
                }
            ),
        }

    except Exception as error:
        print("Error:", str(error))

        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST",
            },
            "body": json.dumps({"success": False, "error": str(error)}),
        }
