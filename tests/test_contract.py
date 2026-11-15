"""验证训练输入与平台一致，且未达到质量门槛时不能发布。"""
import unittest

from knowledge_model_finetuning.prompt import SYSTEM_PROMPT, TEMPLATE_VERSION, build_messages
from knowledge_model_finetuning.evaluation import evaluate, validate_report


class ContractTests(unittest.TestCase):
    """覆盖实际输入模板和发布的保守失败边界。"""

    def test_prompt_matches_platform_and_keeps_review_outside_messages(self):
        """训练与推理使用相同逐字符输入，审核数据不进入消息。"""
        evidence = [{"title": "Python操作", "version": "3.11", "location": "第一章", "content": "使用帮助命令。", "review": "不得进入输入"}]
        messages = build_messages("如何查看帮助？", evidence, "查看帮助。[S1]")
        self.assertEqual(messages, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "问题：如何查看帮助？\n\n编号证据：\n[S1] 文档：Python操作；版本：3.11；出处：第一章\n使用帮助命令。"},
            {"role": "assistant", "content": "查看帮助。[S1]"},
        ])
        self.assertEqual(TEMPLATE_VERSION, "kb-evidence-v1")

    def test_empty_evidence_remains_explicit(self):
        """拒答样例保留空证据区，不伪造来源编号。"""
        self.assertEqual(build_messages("未知问题", [])[-1]["content"], "问题：未知问题\n\n编号证据：\n")

    def test_no_evaluation_samples_cannot_publish(self):
        """空测试集不能生成看似合格的报告。"""
        with self.assertRaises(ValueError):
            validate_report(evaluate([]))

    def test_handwritten_pass_flag_cannot_publish(self):
        """只有通过标志和比例而无真实分子分母时拒绝发布。"""
        with self.assertRaises(ValueError):
            validate_report({"passed": True, "metrics": {"base": {}, "finetuned": {}}})
