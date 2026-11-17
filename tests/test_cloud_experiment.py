"""回归云端语料正文边界和自动拒答代理的消费者可见错误。"""
import importlib.util
from pathlib import Path
import sys
import os
import signal
import socket
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cloud_experiment", ROOT / "scripts/prepare_experiment.py")
EXPERIMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPERIMENT)
sys.modules["prepare_experiment"] = EXPERIMENT
EXTRACTIVE_SPEC = importlib.util.spec_from_file_location("cloud_extractive", ROOT / "scripts/prepare_extractive.py")
EXTRACTIVE = importlib.util.module_from_spec(EXTRACTIVE_SPEC)
EXTRACTIVE_SPEC.loader.exec_module(EXTRACTIVE)
DEPLOY_SPEC = importlib.util.spec_from_file_location("cloud_deploy", ROOT / "scripts/verify_deployment.py")
DEPLOY = importlib.util.module_from_spec(DEPLOY_SPEC)
DEPLOY_SPEC.loader.exec_module(DEPLOY)


class CloudExperimentTests(unittest.TestCase):
    """验证技术正文不含站点反馈，技术能力描述不被误判为拒答。"""

    @unittest.skipUnless(importlib.util.find_spec("bs4"), "正文提取需要可选beautifulsoup4")
    def test_official_feedback_is_not_training_evidence(self):
        """复现官方Docsy把反馈控件嵌在正文容器内的页面结构。"""
        from bs4 import BeautifulSoup
        technical = "Pod中的容器共享网络命名空间，应用日志用于观察应用内部状态及排查运行问题。" * 24
        html = ("<div class='td-content'><h1>Pod运行诊断</h1><p>" + technical + "</p>"
                "<pre>kubectl get pods</pre><div id='pre-footer'>"
                "<p class='feedback--response feedback--response__hidden'>否，感谢反馈。</p>"
                "<a href='https://stackoverflow.com/'>Stack Overflow</a></div></div>")
        text, headings = EXPERIMENT.extract_body(BeautifulSoup(html, "html.parser"), False)
        self.assertIn(technical, text)
        self.assertIn("kubectl get pods", text)
        self.assertEqual(headings, ["Pod运行诊断"])
        self.assertNotIn("感谢反馈", text)
        self.assertNotIn("Stack Overflow", text)

    def test_technical_insufficiency_is_not_answer_refusal(self):
        """区分源文档讨论功能边界与当前回答明确拒绝推断。"""
        self.assertFalse(EXPERIMENT.refusal_proxy("原生功能通常不足以构成完整的日志记录方案。[S1]"))
        self.assertFalse(EXPERIMENT.refusal_proxy("其好处不足以抵销增加的实现复杂度。[S1]"))
        self.assertTrue(EXPERIMENT.refusal_proxy("现有证据不足以确定具体原因，无法判断当前故障。[S1]"))
        self.assertFalse(EXPERIMENT.refusal_proxy("当 CPU 的数量无法确定时，会引发 NotImplementedError。[S1]"))

    def test_formatted_source_identifier_remains_a_valid_question(self):
        """代码标识符的自然问法不应因原文换行和回引号被拒绝。"""
        plan = {"refusal_type": None, "evidence": [], "reference_construction": {
            "excerpts": ["类属性 \n`C.z`\n 将是 \n`10`\n，类属性 `C.t` 将是 `20`。"]}}
        terms = EXTRACTIVE.question_source_terms("C.z 与 C.t 的值分别是多少？", plan)
        self.assertIn("c.z", terms[0])
        self.assertIn("c.t", terms[0])

    def test_irrelevant_evidence_cannot_supply_the_question_topic(self):
        """无关题必须来自独立出题上下文，不能反过来围绕所给证据提问。"""
        plan = {"refusal_type": "irrelevant", "evidence": [{"content": "Pod中的容器共享网络命名空间。"}],
                "reference_construction": {"excerpts": ["Python 异常处理使用 try 和 except 语句。"]}}
        with self.assertRaises(ValueError):
            EXTRACTIVE.question_source_terms("Pod中的容器怎么共享网络命名空间？", plan)

    def test_questions_can_match_mathematical_and_date_literals(self):
        """公式和中文日期中的数字也是可核验事实，不能仅识别单词术语。"""
        expression = {"refusal_type": None, "evidence": [], "reference_construction": {
            "excerpts": ["`-3**2` 会被解释成 `-(3**2)`，结果是 `-9`。"]}}
        self.assertIn("-3**2", EXTRACTIVE.question_source_terms("`-3**2`为何等于-9？", expression)[0])
        dates = {"refusal_type": None, "evidence": [], "reference_construction": {
            "excerpts": ["于2003年12月29日星期一开始，至2004年1月4日星期日结束。"]}}
        self.assertIn("2003", EXTRACTIVE.question_source_terms("2003年12月29日至2004年1月4日是何时？", dates)[0])

    @unittest.skipUnless(os.name == "posix", "云端服务使用POSIX进程组")
    def test_stopping_launcher_also_releases_child_server_port(self):
        """启动真实父子HTTP服务，停止时必须连同子进程释放监听端口。"""
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        launcher = ("import subprocess,sys; subprocess.run([sys.executable,'-m','http.server',"
                    "sys.argv[1],'--bind','127.0.0.1'],check=True)")
        process = subprocess.Popen([sys.executable, "-c", launcher, str(port)], start_new_session=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 5
            while not DEPLOY.port_is_open(port) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(DEPLOY.port_is_open(port), "真实子服务没有启动")
            DEPLOY.stop_server_group(process, port)
            self.assertIsNotNone(process.poll())
            self.assertFalse(DEPLOY.port_is_open(port), "只停止了启动器，子服务仍占用端口")
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
