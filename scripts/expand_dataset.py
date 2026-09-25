#!/usr/bin/env python3
"""
Generate and expand sufficiency evaluation benchmark dataset from N=54 to N=108.
Maintains perfect balance:
- 36 Positive (label: 1)
- 36 Near-Miss (label: 0)
- 36 Unanswerable (label: 0)
"""

import json
from pathlib import Path

DATASET_PATH = Path("benchmarks/datasets/sufficiency_eval.json")

# 18 new triplets (18 positive, 18 near_miss, 18 unanswerable) = 54 items
NEW_ITEMS = [
    # --- Cloud & DevOps (6 triplets) ---
    # 1. K8s OOMKilled
    {
        "id": "pos_k8s_oom",
        "query": "KubernetesでPodがOOMKilled（Exit Code 137）で終了した場合の対処法は？",
        "document": "PodがExit Code 137で終了しOOMKilledと表示された場合、コンテナがメモリ制限（resources.limits.memory）を超過したことを示します。対処法として、マニフェスト内のlimits.memoryの値を引き上げるか、アプリケーションのメモリリークをプロファイリングして解消します。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_k8s_oom",
        "query": "KubernetesでPodがOOMKilled（Exit Code 137）で終了した場合の対処法は？",
        "document": "KubernetesのPodはコンテナの最小実行単位です。PodのライフサイクルにはPending、Running、Succeeded、Failed、Unknownがあり、コンテナが異常終了するとRestartPolicyに従って再起動が試行されます。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_k8s_oom",
        "query": "KubernetesでPodがOOMKilled（Exit Code 137）で終了した場合の対処法は？",
        "document": "Docker Composeは複数コンテナの定義と実行を行うツールです。docker-compose.ymlファイルを使用してサービス、ネットワーク、ボリュームを一括で設定できます。",
        "label": 0,
        "type": "unanswerable",
    },

    # 2. AWS IAM AssumeRole
    {
        "id": "pos_aws_assume",
        "query": "AWS CLIで一時認証情報を取得するsts:AssumeRoleのコマンド例は？",
        "document": "AWS CLIで一時認証情報を取得するには、'aws sts assume-role --role-arn arn:aws:iam::123456789012:role/DeployRole --role-session-name DeploySession' を実行します。返却されるAccessKeyId、SecretAccessKey、SessionTokenを環境変数に設定します。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_aws_assume",
        "query": "AWS CLIで一時認証情報を取得するsts:AssumeRoleのコマンド例は？",
        "document": "AWS Identity and Access Management (IAM) は、AWSリソースへのアクセスを安全に管理するためのサービスです。IAMロールを使用することで、長期的な認証情報を共有することなく安全な権限委譲が可能です。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_aws_assume",
        "query": "AWS CLIで一時認証情報を取得するsts:AssumeRoleのコマンド例は？",
        "document": "Amazon S3は高い耐久性とスケーラビリティを提供するオブジェクトストレージサービスです。バケットポリシーを設定してアクセス権限を制御します。",
        "label": 0,
        "type": "unanswerable",
    },

    # 3. Nginx 502 Bad Gateway
    {
        "id": "pos_nginx_502",
        "query": "Nginxで502 Bad Gatewayが発生した際の主な原因と確認すべきログは？",
        "document": "Nginxの502 Bad Gatewayは、リバースプロキシ先の上流サーバー（GunicornやNode.js等）が停止しているか、ソケット通信が拒否された場合に発生します。確認すべきログは /var/log/nginx/error.log およびバックエンドのログ（例: gunicorn.log）です。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_nginx_502",
        "query": "Nginxで502 Bad Gatewayが発生した際の主な原因と確認すべきログは？",
        "document": "Nginxは高性能なHTTPサーバーおよびリバースプロキシサーバーです。設定ファイルは主に /etc/nginx/nginx.conf に配置され、バーチャルホストの設定は sites-available ディレクトリで管理されます。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_nginx_502",
        "query": "Nginxで502 Bad Gatewayが発生した際の主な原因と確認すべきログは？",
        "document": "Apache HTTP Serverは長年にわたり広く利用されているオープンソースのWebサーバーです。.htaccessファイルを用いてディレクトリ単位での設定オーバーライドが可能です。",
        "label": 0,
        "type": "unanswerable",
    },

    # 4. SSH Permission denied (publickey)
    {
        "id": "pos_ssh_perm",
        "query": "SSH接続時にPermission denied (publickey)となる場合のパーミッション確認手順は？",
        "document": "SSHでPermission denied (publickey)が発生した場合、クライアント側の秘密鍵 (~/.ssh/id_rsa) のパーミッションが600、~/.ssh ディレクトリが700であるか確認します。またサーバー側の ~/.ssh/authorized_keys が600、~/.ssh が700であることを確認してください。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_ssh_perm",
        "query": "SSH接続時にPermission denied (publickey)となる場合のパーミッション確認手順は？",
        "document": "Secure Shell (SSH) は暗号化通信を利用してリモートホストに安全にアクセスするためのプロトコルです。デフォルトではTCPポート22番を使用し、公開鍵認証やパスワード認証をサポートしています。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_ssh_perm",
        "query": "SSH接続時にPermission denied (publickey)となる場合のパーミッション確認手順は？",
        "document": "FTPプロトコルは平文で認証情報を送信するため、機密性の高い通信には適していません。ファイル転送にはSFTPまたはFTPSの使用が推奨されます。",
        "label": 0,
        "type": "unanswerable",
    },

    # 5. PostgreSQL too many connections
    {
        "id": "pos_pg_conn",
        "query": "PostgreSQLで'FATAL: remaining connection slots are reserved'が発生した時のmax_connections設定方法は？",
        "document": "PostgreSQLで接続上限に達した場合、postgresql.conf 内の max_connections（デフォルト100）の値を増やしてサーバーを再起動します。本番環境では接続数の無制限な増加を防ぐため、PgBouncer等のコネクションプーラーの導入を推奨します。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_pg_conn",
        "query": "PostgreSQLで'FATAL: remaining connection slots are reserved'が発生した時のmax_connections設定方法は？",
        "document": "PostgreSQLは高度な機能を備えたオープンソースのリレーショナルデータベース管理システムです。ACID特性を完全にサポートし、複雑なクエリや外部キー制約、トリガーを処理できます。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_pg_conn",
        "query": "PostgreSQLで'FATAL: remaining connection slots are reserved'が発生した時のmax_connections設定方法は？",
        "document": "Redisはインメモリデータ構造ストアであり、データベース、キャッシュ、メッセージブローカーとして利用されます。キー・バリュー形式で超高速な読み書きを実現します。",
        "label": 0,
        "type": "unanswerable",
    },

    # 6. Docker build cache
    {
        "id": "pos_docker_cache",
        "query": "Dockerfileでキャッシュ効率を最大化するRUN/COPYの順序設計は？",
        "document": "Dockerfileでは変更頻度の低い命令を前方に配置します。具体的には、ソースコード全体をコピーする前に、パッケージ定義（package.jsonやpyproject.toml）のみをCOPYし、RUN npm install や RUN uv sync を実行した後に、アプリケーションコード（COPY . .）を配置します。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_docker_cache",
        "query": "Dockerfileでキャッシュ効率を最大化するRUN/COPYの順序設計は？",
        "document": "Dockerイメージはレイヤー構造になっており、各命令ごとに新しいレイヤーが作成されます。イメージサイズを削減するためにマルチステージビルドが広く活用されています。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_docker_cache",
        "query": "Dockerfileでキャッシュ効率を最大化するRUN/COPYの順序設計は？",
        "document": "Gitは分散型バージョン管理システムであり、複数の開発者が同時にコードを変更・マージすることを可能にします。ブランチ戦略としてGit-flowやGitHub Flowがあります。",
        "label": 0,
        "type": "unanswerable",
    },

    # --- Hardware & Technical Identifiers (6 triplets) ---
    # 7. NVIDIA H100 PCIe vs SXM5
    {
        "id": "pos_h100_spec",
        "query": "NVIDIA H100 TensorコアGPUのPCIe版とSXM5版のTDP（消費電力）の違いは？",
        "document": "NVIDIA公式仕様表によれば、H100 PCIe版の最大消費電力（TDP）は350W（空冷・パッシブ）であるのに対し、H100 SXM5版のTDPは最大700W（液冷または高風量空冷）です。SXM5版はNVLink帯域も900GB/sとPCIe版より高速です。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_h100_spec",
        "query": "NVIDIA H100 TensorコアGPUのPCIe版とSXM5版のTDP（消費電力）の違いは？",
        "document": "NVIDIA H100はHopperアーキテクチャを採用したエンタープライズ向けGPUです。Transformer Engineを搭載し、FP8精度でのAI学習・推論を大幅に加速します。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_h100_spec",
        "query": "NVIDIA H100 TensorコアGPUのPCIe版とSXM5版のTDP（消費電力）の違いは？",
        "document": "NVIDIA GeForce RTX 4090はAda Lovelaceアーキテクチャを採用したフラッグシップゲーミングGPUです。24GBのGDDR6Xメモリを搭載しています。",
        "label": 0,
        "type": "unanswerable",
    },

    # 8. Dell PowerEdge R750 vs R740
    {
        "id": "pos_dell_r750",
        "query": "Dell PowerEdge R750サーバーがサポートするPCIeバスの規格は？",
        "document": "Dell EMC PowerEdge R750の仕様書によると、第3世代インテルXeonスケーラブル・プロセッサーを搭載し、最大8つのPCIe Gen 4スロットをサポートしています。前世代のR740（PCIe Gen 3）から転送帯域が2倍に向上しています。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_dell_r750",
        "query": "Dell PowerEdge R750サーバーがサポートするPCIeバスの規格は？",
        "document": "Dell EMC PowerEdge R750は、高いパフォーマンスと拡張性を備えた2Uラックマウント型デュアルソケットサーバーです。仮想化、データベース、ハイパフォーマンスコンピューティングに適しています。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_dell_r750",
        "query": "Dell PowerEdge R750サーバーがサポートするPCIeバスの規格は？",
        "document": "Synology NASは直感的なDiskStation Manager（DSM）OSを搭載したネットワーク接続ストレージです。RAIDアレイの構築やファイル共有が容易に行えます。",
        "label": 0,
        "type": "unanswerable",
    },

    # 9. Cisco Catalyst 9300 Uplink
    {
        "id": "pos_cisco_9300",
        "query": "Cisco Catalyst 9300シリーズで利用可能なネットワークモジュールC9300-NM-8Xのポート仕様は？",
        "document": "Ciscoデータシートによると、C9300-NM-8Xモジュールは8個の10G SFP+スロットを提供します。すべてのポートで1G SFPまたは10G SFP+トランシーバをサポートし、アップリンク帯域を拡張できます。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_cisco_9300",
        "query": "Cisco Catalyst 9300シリーズで利用可能なネットワークモジュールC9300-NM-8Xのポート仕様は？",
        "document": "Cisco Catalyst 9300シリーズスイッチは、セキュリティ、IoT、モビリティ、クラウド向けに構築されたCiscoの代表的なスタッカブルエンタープライズスイッチプラットフォームです。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_cisco_9300",
        "query": "Cisco Catalyst 9300シリーズで利用可能なネットワークモジュールC9300-NM-8Xのポート仕様は？",
        "document": "ヤマハルーターRTX1220は中小規模拠点向けのVPNルーターです。全ポートギガビットイーサネットに対応し、ISDN回線収容機能も備えています。",
        "label": 0,
        "type": "unanswerable",
    },

    # 10. DDR5-5600 vs DDR4-3200
    {
        "id": "pos_ddr5_spec",
        "query": "DDR5-5600メモリの標準動作電圧と理論最大転送速度（単一チャネル）は？",
        "document": "JEDEC規格によれば、DDR5メモリの標準動作電圧は1.1V（DDR4の1.2Vから低減）です。またDDR5-5600の単一64ビットチャネルあたりの理論最大転送帯域幅は44.8 GB/sです。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_ddr5_spec",
        "query": "DDR5-5600メモリの標準動作電圧と理論最大転送速度（単一チャネル）は？",
        "document": "DDR5 SDRAMはパソコンやサーバー用のメインメモリ規格です。DDR4と比較して電力効率が向上し、On-Die ECCによる信頼性向上や2系統の独立32ビットサブチャネル構造が導入されています。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_ddr5_spec",
        "query": "DDR5-5600メモリの標準動作電圧と理論最大転送速度（単一チャネル）は？",
        "document": "SSD（Solid State Drive）はフラッシュメモリを記録媒体とするストレージデバイスです。NVMeプロトコルとPCIeインターフェースにより超高速なデータアクセスを実現します。",
        "label": 0,
        "type": "unanswerable",
    },

    # 11. Wi-Fi 6E frequency
    {
        "id": "pos_wifi6e_freq",
        "query": "Wi-Fi 6E（IEEE 802.11ax拡張）で新たに開放された周波数帯は？",
        "document": "Wi-Fi 6Eでは、従来の2.4GHz帯および5GHz帯に加え、新たに6GHz帯（5.925GHz〜7.125GHz、日本では最大24チャンネル）の周波数帯が利用可能となり、電波干渉のない広帯域通信が実現されました。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_wifi6e_freq",
        "query": "Wi-Fi 6E（IEEE 802.11ax拡張）で新たに開放された周波数帯は？",
        "document": "Wi-Fi 6（IEEE 802.11ax）はOFDMAや1024-QAM、MU-MIMO技術を採用し、多数の端末が密集する混雑環境でも安定した高速通信を提供できる無線LAN規格です。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_wifi6e_freq",
        "query": "Wi-Fi 6E（IEEE 802.11ax拡張）で新たに開放された周波数帯は？",
        "document": "Bluetooth 5.0は近距離無線通信規格であり、BLE（Bluetooth Low Energy）の通信速度が2Mbpsに倍増し、通信範囲も4倍に拡大されました。",
        "label": 0,
        "type": "unanswerable",
    },

    # 12. USB4 vs Thunderbolt 4
    {
        "id": "pos_usb4_tb4",
        "query": "USB4 40GbpsとThunderbolt 4の最低要件におけるPCIeデータ転送速度の違いは？",
        "document": "Thunderbolt 4仕様では最低32GbpsのPCIeデータ転送（外部GPUや高速SSD用）が必須要件として義務付けられていますが、標準のUSB4仕様ではPCIeトンネリング機能自体がオプションとなっており、転送速度の必須保証はありません。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_usb4_tb4",
        "query": "USB4 40GbpsとThunderbolt 4の最低要件におけるPCIeデータ転送速度の違いは？",
        "document": "USB Type-Cコネクタは表裏の区別なく挿抜可能な汎用端子です。USB Power Delivery（USB PD）による最大240Wの給電やDisplayPort Alternate Modeによる映像出力に対応しています。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_usb4_tb4",
        "query": "USB4 40GbpsとThunderbolt 4の最低要件におけるPCIeデータ転送速度の違いは？",
        "document": "HDMI 2.1はテレビやモニター向けの映像伝送規格です。最大48Gbpsの帯域幅を持ち、非圧縮8K 60Hzや4K 120Hzの出力、可変リフレッシュレート（VRR）に対応します。",
        "label": 0,
        "type": "unanswerable",
    },

    # --- Corporate HR & Governance (3 triplets) ---
    # 13. 慶弔休暇
    {
        "id": "pos_corp_condolence",
        "query": "就業規則における本人の結婚に伴う特別有給休暇（慶弔休暇）の日数は？",
        "document": "就業規則第28条（慶弔休暇）に基づき、社員本人が結婚した場合は、連続した5営業日の特別有給休暇を取得することができます。なお結婚の日から起算して6ヶ月以内に取得する必要があります。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_corp_condolence",
        "query": "就業規則における本人の結婚に伴う特別有給休暇（慶弔休暇）の日数は？",
        "document": "当社はワークライフバランスを重視し、有給休暇の取得促進に努めています。年次有給休暇は入社後半年で10日付与され、勤続年数に応じて最大年間20日まで付与されます。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_corp_condolence",
        "query": "就業規則における本人の結婚に伴う特別有給休暇（慶弔休暇）の日数は？",
        "document": "当社のオフィスセキュリティ規定では、入退館時にIC社員証のタッチが義務付けられています。来客時には必ず受付表に記入の上、ゲストカードを貸与します。",
        "label": 0,
        "type": "unanswerable",
    },

    # 14. 副業許可
    {
        "id": "pos_corp_sidejob",
        "query": "副業兼業ガイドラインにおいて提出が必要な申請書と承認基準は？",
        "document": "副業兼業ガイドライン第3条に基づき、副業を希望する社員は開始日の2週間前までに『副業・兼業許可申請書』を人事部へ提出する必要があります。本業の利益相反にならないこと、過重労働（月40時間を超えないこと）が承認の基準です。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_corp_sidejob",
        "query": "副業兼業ガイドラインにおいて提出が必要な申請書と承認基準は？",
        "document": "社員は就業時間中は職務に専念する義務を負います。業務上の秘密情報や顧客情報を社外に漏洩することは固く禁じられており、違反した場合は懲戒処分の対象となります。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_corp_sidejob",
        "query": "副業兼業ガイドラインにおいて提出が必要な申請書と承認基準は？",
        "document": "定期健康診断は労働安全衛生法に基づき年1回受診が義務付けられています。35歳以上の社員については生活習慣病予防健診または人間ドックを受診できます。",
        "label": 0,
        "type": "unanswerable",
    },

    # 15. 在宅勤務手当
    {
        "id": "pos_corp_remote",
        "query": "リモートワーク勤務規程における在宅勤務手当の支給額と対象条件は？",
        "document": "在宅勤務規程第12条により、月の所定労働日数の半分（50%）以上をテレワークで勤務した社員に対し、光熱費・通信環境補助として月額5,000円の在宅勤務手当が支給されます。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_corp_remote",
        "query": "リモートワーク勤務規程における在宅勤務手当の支給額と対象条件は？",
        "document": "当社のテレワーク制度は柔軟な働き方を支援する目的で導入されました。セキュリティ確保のため、会社貸与のPCおよびVPN接続を利用して業務を行う必要があります。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_corp_remote",
        "query": "リモートワーク勤務規程における在宅勤務手当の支給額と対象条件は？",
        "document": "通勤交通費は最も経済的かつ合理的な経路に基づき月額上限5万円まで全額支給されます。定期券代は給与支給日に合わせて支給されます。",
        "label": 0,
        "type": "unanswerable",
    },

    # --- Legal & Security (3 triplets) ---
    # 16. 個人情報漏洩報告
    {
        "id": "pos_legal_leak",
        "query": "個人情報保護法における個人データ漏洩発生時の個人情報保護委員会への報告期限は？",
        "document": "改正個人情報保護法第26条および施行規則に基づき、漏洩発生を把握した日から速報はおよそ3日以内、確報は30日以内（不正アクセス等の場合は60日以内）に個人情報保護委員会への報告が義務付けられています。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_legal_leak",
        "query": "個人情報保護法における個人データ漏洩発生時の個人情報保護委員会への報告期限は？",
        "document": "個人情報保護法は個人の権利利益を保護することを目的とした法律です。事業者は個人情報を取得する際には利用目的を特定・公表し、安全管理措置を講じる必要があります。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_legal_leak",
        "query": "個人情報保護法における個人データ漏洩発生時の個人情報保護委員会への報告期限は？",
        "document": "労働基準法第36条に基づく時間外・休日労働に関する協定（36協定）を締結し、所轄の労働基準監督署長に届け出なければ、法定労働時間を超える労働を命じることはできません。",
        "label": 0,
        "type": "unanswerable",
    },

    # 17. NDA 秘密保持期間
    {
        "id": "pos_legal_nda",
        "query": "標準秘密保持契約（NDA）における契約終了後の秘密保持義務の有効存続期間は？",
        "document": "秘密保持契約書第8条（存続条項）によれば、本契約が終了した場合であっても、本契約に基づき開示された秘密情報に関する秘密保持義務は、契約終了の日から5年間有効に存続するものとします。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_legal_nda",
        "query": "標準秘密保持契約（NDA）における契約終了後の秘密保持義務の有効存続期間は？",
        "document": "秘密保持契約（NDA）は業務提携や共同研究の検討にあたり開示される機密情報を保護するための合意文書です。開示目的以外の使用禁止や第三者への漏洩防止が定められます。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_legal_nda",
        "query": "標準秘密保持契約（NDA）における契約終了後の秘密保持義務の有効存続期間は？",
        "document": "著作権法において、著作権は著作物の創作時に自動的に発生し、保護期間は著作者の死後70年（法人著作の場合は公表後70年）存続します。",
        "label": 0,
        "type": "unanswerable",
    },

    # 18. パスワードポリシー & MFA
    {
        "id": "pos_sec_mfa",
        "query": "社内セキュリティ基準における管理特権アカウントのパスワード最小文字数と認証要件は？",
        "document": "情報セキュリティ基本規程第15条によると、管理者特権（Administrator / root）アカウントのパスワードは英大文字・小文字・数字・記号を混在させた最低16文字以上とし、FIDO2またはTOTPによる多要素認証（MFA）の設定が必須です。",
        "label": 1,
        "type": "positive",
    },
    {
        "id": "neg_sec_mfa",
        "query": "社内セキュリティ基準における管理特権アカウントのパスワード最小文字数と認証要件は？",
        "document": "情報システムへのログインには各個人に割り当てられた一意のユーザーIDを使用します。アカウントの共用は禁止されており、定期的な利用状況の監査が実施されます。",
        "label": 0,
        "type": "near_miss",
    },
    {
        "id": "unans_sec_mfa",
        "query": "社内セキュリティ基準における管理特権アカウントのパスワード最小文字数と認証要件は？",
        "document": "防火管理者は消防法に基づき建物の防火管理業務を遂行する責任者です。定期的な消防訓練の実施や消防用設備の点検・報告を行います。",
        "label": 0,
        "type": "unanswerable",
    },
]

def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)
    print(f"Existing dataset items: {len(existing)}")

    # Append new items
    expanded = existing + NEW_ITEMS
    print(f"Expanded dataset items: {len(expanded)}")

    counts = {}
    for item in expanded:
        t = item.get("type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    print(f"New breakdown: {counts}")

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(expanded, f, ensure_ascii=False, indent=2)
    print(f"Saved expanded dataset to {DATASET_PATH}")

if __name__ == "__main__":
    main()
