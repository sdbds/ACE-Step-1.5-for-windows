"""
ACE-Step OpenRouter API 压力测试脚本

支持并发测试，用于测试服务的最大 QPS 和性能表现。

Usage:
    # 基本测试 - 10 并发，100 请求
    python -m openrouter.stress_test

    # 自定义参数测试
    python -m openrouter.stress_test --concurrency 50 --requests 500

    # 逐步加压测试
    python -m openrouter.stress_test --mode ramp --max-concurrency 100 --step 10

    # 持续压测
    python -m openrouter.stress_test --mode duration --duration 60 --concurrency 20
"""

import argparse
import json
import os
import sys
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import defaultdict
from datetime import datetime
import queue

import requests


# =============================================================================
# 配置
# =============================================================================

DEFAULT_BASE_URL = "https://api.acemusic.ai"
DEFAULT_CONCURRENCY = 4
DEFAULT_TOTAL_REQUESTS = 100


@dataclass
class RequestResult:
    """单次请求结果"""
    success: bool
    status_code: int
    latency: float  # 秒
    error_message: str = ""
    timestamp: float = 0.0


@dataclass
class StressTestStats:
    """压力测试统计数据"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies: List[float] = field(default_factory=list)
    errors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    status_codes: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    start_time: float = 0.0
    end_time: float = 0.0

    def add_result(self, result: RequestResult):
        """添加请求结果"""
        self.total_requests += 1
        self.status_codes[result.status_code] += 1

        if result.success:
            self.successful_requests += 1
            self.latencies.append(result.latency)
        else:
            self.failed_requests += 1
            self.errors[result.error_message] += 1

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def duration(self) -> float:
        """测试持续时间"""
        return self.end_time - self.start_time

    @property
    def qps(self) -> float:
        """每秒请求数 (QPS)"""
        if self.duration == 0:
            return 0.0
        return self.total_requests / self.duration

    @property
    def successful_qps(self) -> float:
        """成功请求的 QPS"""
        if self.duration == 0:
            return 0.0
        return self.successful_requests / self.duration

    def get_latency_stats(self) -> Dict[str, float]:
        """获取延迟统计数据"""
        if not self.latencies:
            return {
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
                "median": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)

        return {
            "min": min(sorted_latencies),
            "max": max(sorted_latencies),
            "avg": statistics.mean(sorted_latencies),
            "median": statistics.median(sorted_latencies),
            "p90": sorted_latencies[int(n * 0.90)] if n > 0 else 0.0,
            "p95": sorted_latencies[int(n * 0.95)] if n > 0 else 0.0,
            "p99": sorted_latencies[int(n * 0.99)] if n > 0 else 0.0,
        }


class StressTester:
    """压力测试器"""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 300,
        test_type: str = "health",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.test_type = test_type
        self.session = requests.Session()
        self.lock = threading.Lock()
        self.request_counter = 0
        self.live_stats = StressTestStats()

    def get_headers(self) -> dict:
        """构建请求头"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def make_request(self) -> RequestResult:
        """执行单次请求"""
        start_time = time.time()
        timestamp = start_time

        try:
            if self.test_type == "health":
                resp = requests.get(
                    f"{self.base_url}/health",
                    timeout=self.timeout
                )
            elif self.test_type == "models":
                resp = requests.get(
                    f"{self.base_url}/api/v1/models",
                    headers=self.get_headers(),
                    timeout=self.timeout
                )
            elif self.test_type == "generate":
                payload = self._get_generate_payload()
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.get_headers(),
                    json=payload,
                    timeout=self.timeout
                )
            elif self.test_type == "instrumental":
                payload = self._get_instrumental_payload()
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.get_headers(),
                    json=payload,
                    timeout=self.timeout
                )
            else:
                # 默认 health
                resp = requests.get(
                    f"{self.base_url}/health",
                    timeout=self.timeout
                )

            latency = time.time() - start_time

            if resp.status_code == 200:
                return RequestResult(
                    success=True,
                    status_code=resp.status_code,
                    latency=latency,
                    timestamp=timestamp
                )
            else:
                return RequestResult(
                    success=False,
                    status_code=resp.status_code,
                    latency=latency,
                    error_message=f"HTTP {resp.status_code}",
                    timestamp=timestamp
                )

        except requests.exceptions.Timeout:
            return RequestResult(
                success=False,
                status_code=0,
                latency=time.time() - start_time,
                error_message="Timeout",
                timestamp=timestamp
            )
        except requests.exceptions.ConnectionError as e:
            return RequestResult(
                success=False,
                status_code=0,
                latency=time.time() - start_time,
                error_message=f"ConnectionError: {str(e)[:50]}",
                timestamp=timestamp
            )
        except Exception as e:
            return RequestResult(
                success=False,
                status_code=0,
                latency=time.time() - start_time,
                error_message=f"{type(e).__name__}: {str(e)[:50]}",
                timestamp=timestamp
            )

    def _get_generate_payload(self) -> dict:
        """获取生成请求的 payload"""
        return {
            "messages": [
                {"role": "user", "content": "Generate an upbeat pop song about summer"}
            ],
            "sample_mode": True,
            "audio_config": {
                "vocal_language": "en",
                "duration": 30,
            },
        }

    def _get_instrumental_payload(self) -> dict:
        """获取纯音乐请求的 payload"""
        return {
            "messages": [
                {"role": "user", "content": "<prompt>Epic orchestral cinematic score</prompt>"}
            ],
            "audio_config": {
                "instrumental": True,
                "duration": 30,
            },
        }

    def run_fixed_requests(
        self,
        concurrency: int,
        total_requests: int,
        show_progress: bool = True
    ) -> StressTestStats:
        """固定请求数模式"""
        stats = StressTestStats()
        stats.start_time = time.time()

        completed = 0
        completed_lock = threading.Lock()

        def worker():
            nonlocal completed
            result = self.make_request()

            with completed_lock:
                completed += 1
                stats.add_result(result)

                if show_progress and completed % 10 == 0:
                    elapsed = time.time() - stats.start_time
                    current_qps = completed / elapsed if elapsed > 0 else 0
                    print(
                        f"\r进度: {completed}/{total_requests} "
                        f"({completed/total_requests*100:.1f}%) | "
                        f"成功: {stats.successful_requests} | "
                        f"失败: {stats.failed_requests} | "
                        f"当前 QPS: {current_qps:.2f}",
                        end="", flush=True
                    )

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker) for _ in range(total_requests)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"\n工作线程异常: {e}")

        stats.end_time = time.time()

        if show_progress:
            print()  # 换行

        return stats

    def run_duration_based(
        self,
        concurrency: int,
        duration: int,
        show_progress: bool = True
    ) -> StressTestStats:
        """持续时间模式"""
        stats = StressTestStats()
        stats.start_time = time.time()
        stop_event = threading.Event()

        def worker():
            while not stop_event.is_set():
                result = self.make_request()
                with self.lock:
                    stats.add_result(result)

        # 启动工作线程
        threads = []
        for _ in range(concurrency):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)

        # 显示进度
        try:
            end_time = time.time() + duration
            while time.time() < end_time:
                elapsed = time.time() - stats.start_time
                remaining = duration - elapsed
                current_qps = stats.total_requests / elapsed if elapsed > 0 else 0

                if show_progress:
                    print(
                        f"\r剩余时间: {remaining:.1f}s | "
                        f"请求数: {stats.total_requests} | "
                        f"成功: {stats.successful_requests} | "
                        f"失败: {stats.failed_requests} | "
                        f"QPS: {current_qps:.2f}",
                        end="", flush=True
                    )
                time.sleep(0.5)
        finally:
            stop_event.set()

        # 等待所有线程结束
        for t in threads:
            t.join(timeout=5)

        stats.end_time = time.time()

        if show_progress:
            print()  # 换行

        return stats

    def run_ramp_up(
        self,
        max_concurrency: int,
        step: int,
        requests_per_step: int,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """逐步加压模式"""
        results = []

        for concurrency in range(step, max_concurrency + 1, step):
            print(f"\n{'='*60}")
            print(f"测试并发数: {concurrency}")
            print("=" * 60)

            stats = self.run_fixed_requests(
                concurrency=concurrency,
                total_requests=requests_per_step,
                show_progress=show_progress
            )

            latency_stats = stats.get_latency_stats()

            result = {
                "concurrency": concurrency,
                "total_requests": stats.total_requests,
                "successful_requests": stats.successful_requests,
                "failed_requests": stats.failed_requests,
                "success_rate": stats.success_rate,
                "qps": stats.qps,
                "successful_qps": stats.successful_qps,
                "avg_latency": latency_stats["avg"],
                "p95_latency": latency_stats["p95"],
                "p99_latency": latency_stats["p99"],
            }
            results.append(result)

            self._print_step_summary(result)

            # 短暂休息让服务恢复
            time.sleep(2)

        return results


    def _print_step_summary(self, result: Dict[str, Any]):
        """打印单步测试摘要"""
        print(f"\n并发数 {result['concurrency']} 测试结果:")
        print(f"  总请求数: {result['total_requests']}")
        print(f"  成功/失败: {result['successful_requests']}/{result['failed_requests']}")
        print(f"  成功率: {result['success_rate']:.2f}%")
        print(f"  QPS: {result['qps']:.2f}")
        print(f"  成功 QPS: {result['successful_qps']:.2f}")
        print(f"  平均延迟: {result['avg_latency']*1000:.2f}ms")
        print(f"  P95 延迟: {result['p95_latency']*1000:.2f}ms")
        print(f"  P99 延迟: {result['p99_latency']*1000:.2f}ms")


def print_stats(stats: StressTestStats, title: str = "压力测试结果"):
    """打印统计结果"""
    latency_stats = stats.get_latency_stats()

    print("\n")
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)

    print("\n📊 基本统计")
    print("-" * 40)
    print(f"  总请求数:       {stats.total_requests}")
    print(f"  成功请求数:     {stats.successful_requests}")
    print(f"  失败请求数:     {stats.failed_requests}")
    print(f"  成功率:         {stats.success_rate:.2f}%")

    print("\n⏱️ 时间统计")
    print("-" * 40)
    print(f"  测试持续时间:   {stats.duration:.2f} 秒")
    print(f"  总 QPS:         {stats.qps:.2f}")
    print(f"  成功 QPS:       {stats.successful_qps:.2f}")

    print("\n📈 延迟统计 (毫秒)")
    print("-" * 40)
    print(f"  最小延迟:       {latency_stats['min']*1000:.2f}ms")
    print(f"  最大延迟:       {latency_stats['max']*1000:.2f}ms")
    print(f"  平均延迟:       {latency_stats['avg']*1000:.2f}ms")
    print(f"  中位数延迟:     {latency_stats['median']*1000:.2f}ms")
    print(f"  P90 延迟:       {latency_stats['p90']*1000:.2f}ms")
    print(f"  P95 延迟:       {latency_stats['p95']*1000:.2f}ms")
    print(f"  P99 延迟:       {latency_stats['p99']*1000:.2f}ms")

    if stats.status_codes:
        print("\n📋 状态码分布")
        print("-" * 40)
        for code, count in sorted(stats.status_codes.items()):
            percentage = (count / stats.total_requests) * 100
            print(f"  {code}:  {count:>8} ({percentage:.1f}%)")

    if stats.errors:
        print("\n❌ 错误分布 (Top 10)")
        print("-" * 40)
        sorted_errors = sorted(stats.errors.items(), key=lambda x: x[1], reverse=True)[:10]
        for error, count in sorted_errors:
            percentage = (count / stats.total_requests) * 100
            print(f"  {error[:50]}: {count} ({percentage:.1f}%)")

    print("\n" + "=" * 70)


def print_ramp_summary(results: List[Dict[str, Any]]):
    """打印逐步加压测试的汇总"""
    print("\n")
    print("=" * 90)
    print(" 逐步加压测试汇总")
    print("=" * 90)

    # 表头
    print(f"\n{'并发':>8} | {'请求数':>8} | {'成功率':>8} | {'QPS':>10} | {'成功QPS':>10} | {'平均延迟':>10} | {'P99延迟':>10}")
    print("-" * 90)

    # 数据行
    for r in results:
        print(
            f"{r['concurrency']:>8} | "
            f"{r['total_requests']:>8} | "
            f"{r['success_rate']:>7.1f}% | "
            f"{r['qps']:>10.2f} | "
            f"{r['successful_qps']:>10.2f} | "
            f"{r['avg_latency']*1000:>9.1f}ms | "
            f"{r['p99_latency']*1000:>9.1f}ms"
        )

    print("-" * 90)

    # 找出最佳 QPS
    best_qps = max(results, key=lambda x: x['successful_qps'])
    print(f"\n🏆 最佳成功 QPS: {best_qps['successful_qps']:.2f} (并发数: {best_qps['concurrency']})")

    # 找出延迟瓶颈点（P99 延迟开始急剧上升的点）
    for i in range(1, len(results)):
        if results[i]['p99_latency'] > results[i-1]['p99_latency'] * 2:
            print(f"⚠️  延迟瓶颈点: 并发数 {results[i]['concurrency']} (P99 延迟开始急剧上升)")
            break

    # 找出错误率开始上升的点
    for i, r in enumerate(results):
        if r['success_rate'] < 99:
            print(f"⚠️  稳定性下降点: 并发数 {r['concurrency']} (成功率: {r['success_rate']:.1f}%)")
            break

    print()


def main():
    parser = argparse.ArgumentParser(
        description="ACE-Step OpenRouter API 压力测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 健康检查接口压测
  python -m openrouter.stress_test --test health --concurrency 100 --requests 1000

  # 模型列表接口压测
  python -m openrouter.stress_test --test models --concurrency 50 --requests 500

  # 音乐生成接口压测 (注意: 生成请求较慢)
  python -m openrouter.stress_test --test generate --concurrency 5 --requests 20

  # 逐步加压测试
  python -m openrouter.stress_test --mode ramp --max-concurrency 100 --step 10

  # 持续时间压测 (60秒)
  python -m openrouter.stress_test --mode duration --duration 60 --concurrency 50
        """
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        help=f"API 基础 URL (默认: {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="API 密钥 (可选)"
    )
    parser.add_argument(
        "--test",
        choices=["health", "models", "generate", "instrumental"],
        default="health",
        help="要测试的接口类型 (默认: health)"
    )
    parser.add_argument(
        "--mode",
        choices=["fixed", "duration", "ramp"],
        default="fixed",
        help="测试模式: fixed=固定请求数, duration=持续时间, ramp=逐步加压 (默认: fixed)"
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"并发数 (默认: {DEFAULT_CONCURRENCY})"
    )
    parser.add_argument(
        "--requests", "-n",
        type=int,
        default=DEFAULT_TOTAL_REQUESTS,
        help=f"总请求数 (fixed 模式) (默认: {DEFAULT_TOTAL_REQUESTS})"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=60,
        help="测试持续时间秒数 (duration 模式) (默认: 60)"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=100,
        help="最大并发数 (ramp 模式) (默认: 100)"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="并发增长步长 (ramp 模式) (默认: 10)"
    )
    parser.add_argument(
        "--requests-per-step",
        type=int,
        default=100,
        help="每个步骤的请求数 (ramp 模式) (默认: 100)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="请求超时时间秒数 (默认: 300)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出结果到 JSON 文件"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="减少输出信息"
    )

    args = parser.parse_args()

    # 打印配置信息
    print("=" * 70)
    print(" ACE-Step OpenRouter API 压力测试")
    print("=" * 70)
    print(f"  时间:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Base URL:       {args.base_url}")
    print(f"  API Key:        {'已设置' if args.api_key else '未设置'}")
    print(f"  测试接口:       {args.test}")
    print(f"  测试模式:       {args.mode}")

    if args.mode == "fixed":
        print(f"  并发数:         {args.concurrency}")
        print(f"  总请求数:       {args.requests}")
    elif args.mode == "duration":
        print(f"  并发数:         {args.concurrency}")
        print(f"  持续时间:       {args.duration} 秒")
    elif args.mode == "ramp":
        print(f"  最大并发数:     {args.max_concurrency}")
        print(f"  步长:           {args.step}")
        print(f"  每步请求数:     {args.requests_per_step}")

    print(f"  请求超时:       {args.timeout} 秒")
    print("=" * 70)

    # 创建测试器
    tester = StressTester(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        test_type=args.test
    )

    # 执行测试
    try:
        if args.mode == "fixed":
            print(f"\n开始固定请求数测试 (并发: {args.concurrency}, 请求数: {args.requests})...\n")
            stats = tester.run_fixed_requests(
                concurrency=args.concurrency,
                total_requests=args.requests,
                show_progress=not args.quiet
            )
            print_stats(stats, f"压力测试结果 - {args.test.upper()} 接口")

            # 保存结果
            if args.output:
                latency_stats = stats.get_latency_stats()
                output_data = {
                    "test_type": args.test,
                    "mode": args.mode,
                    "concurrency": args.concurrency,
                    "total_requests": stats.total_requests,
                    "successful_requests": stats.successful_requests,
                    "failed_requests": stats.failed_requests,
                    "success_rate": stats.success_rate,
                    "duration": stats.duration,
                    "qps": stats.qps,
                    "successful_qps": stats.successful_qps,
                    "latency": latency_stats,
                    "status_codes": dict(stats.status_codes),
                    "errors": dict(stats.errors),
                    "timestamp": datetime.now().isoformat()
                }
                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"\n结果已保存到: {args.output}")

        elif args.mode == "duration":
            print(f"\n开始持续时间测试 (并发: {args.concurrency}, 时长: {args.duration}秒)...\n")
            stats = tester.run_duration_based(
                concurrency=args.concurrency,
                duration=args.duration,
                show_progress=not args.quiet
            )
            print_stats(stats, f"压力测试结果 - {args.test.upper()} 接口 ({args.duration}秒)")

        elif args.mode == "ramp":
            print(f"\n开始逐步加压测试 (最大并发: {args.max_concurrency}, 步长: {args.step})...\n")
            results = tester.run_ramp_up(
                max_concurrency=args.max_concurrency,
                step=args.step,
                requests_per_step=args.requests_per_step,
                show_progress=not args.quiet
            )
            print_ramp_summary(results)

            # 保存结果
            if args.output:
                output_data = {
                    "test_type": args.test,
                    "mode": args.mode,
                    "max_concurrency": args.max_concurrency,
                    "step": args.step,
                    "requests_per_step": args.requests_per_step,
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                }
                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"结果已保存到: {args.output}")

    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
