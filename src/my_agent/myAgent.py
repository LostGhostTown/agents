from langchain.agents import create_agent
def send_email(to: str,subject: str, body: str):
    """发送邮件"""#必须要有docstring， 即在工具第一行为当前工具的功能描述和使用时机
    email= {
        "to": to,
        "subject": subject,
        "body": body
    }
    return f"Email sent to {to}"
from my_agent.mylm import qianFan
agent=create_agent(
    model=qianFan,
    tools=[send_email],
    system_prompt="你是一个智能助手，可以帮助用户完成各种任务。当用户需要发送邮件时，请使用send_email工具。",
)