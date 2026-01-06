#!/usr/bin/env python3
"""
训练Session管理工具
用于查看、对比、清理训练sessions

使用方法:
  python session_manager.py list                    # 列出所有sessions
  python session_manager.py view <session_id>       # 查看某个session详情
  python session_manager.py compare <id1> <id2>     # 对比两个sessions
  python session_manager.py clean --keep 5          # 只保留最近5个sessions
  python session_manager.py best                    # 显示最佳session
"""

import os
import sys
import glob
import argparse
from datetime import datetime
import re


class SessionManager:
    """Session管理器"""
    
    def __init__(self, log_dir='./log', ckpt_dir='./ckpt', result_dir='./result'):
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.result_dir = result_dir
    
    def list_sessions(self):
        """列出所有sessions"""
        sessions = {}
        
        # 从log文件提取sessions
        for log_file in glob.glob(f"{self.log_dir}/train_*.log"):
            basename = os.path.basename(log_file)
            match = re.match(r'train_(\d{8}_\d{6})\.log', basename)
            if match:
                session_id = match.group(1)
                sessions[session_id] = {'log': log_file}
        
        # 关联checkpoint文件
        for ckpt_file in glob.glob(f"{self.ckpt_dir}/*_best_acc_*.pth"):
            basename = os.path.basename(ckpt_file)
            match = re.match(r'(\d{8}_\d{6})_best_acc_([\d.]+)_epoch_(\d+)\.pth', basename)
            if match:
                session_id = match.group(1)
                acc = float(match.group(2))
                epoch = int(match.group(3))
                
                if session_id not in sessions:
                    sessions[session_id] = {}
                
                if 'checkpoints' not in sessions[session_id]:
                    sessions[session_id]['checkpoints'] = []
                
                sessions[session_id]['checkpoints'].append({
                    'file': ckpt_file,
                    'acc': acc,
                    'epoch': epoch
                })
        
        # 排序
        for session_id in sessions:
            if 'checkpoints' in sessions[session_id]:
                sessions[session_id]['checkpoints'].sort(key=lambda x: x['acc'], reverse=True)
                sessions[session_id]['best_acc'] = sessions[session_id]['checkpoints'][0]['acc']
                sessions[session_id]['best_epoch'] = sessions[session_id]['checkpoints'][0]['epoch']
        
        return sessions
    
    def print_session_list(self):
        """打印session列表"""
        sessions = self.list_sessions()
        
        if not sessions:
            print("未找到任何训练session")
            return
        
        print("\n" + "="*80)
        print("训练Sessions列表")
        print("="*80)
        print(f"{'Session ID':<18} {'最佳准确率':<12} {'Epoch':<8} {'日志':<10} {'模型':<6}")
        print("-"*80)
        
        # 按时间倒序
        for session_id in sorted(sessions.keys(), reverse=True):
            info = sessions[session_id]
            
            has_log = '✓' if 'log' in info else '✗'
            has_ckpt = '✓' if 'checkpoints' in info else '✗'
            
            if 'best_acc' in info:
                acc = f"{info['best_acc']:.2f}%"
                epoch = str(info['best_epoch'])
            else:
                acc = "N/A"
                epoch = "N/A"
            
            print(f"{session_id:<18} {acc:<12} {epoch:<8} {has_log:<10} {has_ckpt:<6}")
        
        print("="*80 + "\n")
    
    def view_session(self, session_id):
        """查看session详情"""
        print("\n" + "="*80)
        print(f"Session: {session_id}")
        print("="*80 + "\n")
        
        # 1. 日志文件
        log_file = f"{self.log_dir}/train_{session_id}.log"
        if os.path.exists(log_file):
            print(f"📄 日志文件: {log_file}")
            size = os.path.getsize(log_file) / 1024
            print(f"   大小: {size:.1f} KB")
            
            # 提取关键信息
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # 找最佳准确率
                for line in reversed(lines):
                    if "Best test accuracy" in line or "best_test_acc" in line:
                        print(f"   {line.strip()}")
                        break
                
                # 找训练时间
                for line in reversed(lines):
                    if "Total time" in line:
                        print(f"   {line.strip()}")
                        break
        else:
            print(f"❌ 日志文件不存在")
        
        # 2. Checkpoint文件
        print(f"\n💾 Checkpoint文件:")
        ckpt_files = glob.glob(f"{self.ckpt_dir}/{session_id}_*.pth")
        
        if ckpt_files:
            for f in sorted(ckpt_files, reverse=True):
                basename = os.path.basename(f)
                size = os.path.getsize(f) / (1024**2)
                
                # 提取准确率
                match = re.search(r'acc_([\d.]+)', basename)
                if match:
                    acc = match.group(1)
                    print(f"   - {basename} ({size:.1f} MB) ⭐ {acc}%")
                else:
                    print(f"   - {basename} ({size:.1f} MB)")
        else:
            print(f"   ❌ 未找到checkpoint文件")
        
        # 3. 结果文件（如果在同一目录）
        print(f"\n📊 结果文件:")
        result_files = glob.glob(f"{self.result_dir}/*.*")
        if result_files:
            for rf in result_files[:5]:  # 只显示前5个
                print(f"   - {os.path.basename(rf)}")
            if len(result_files) > 5:
                print(f"   ... 共 {len(result_files)} 个文件")
        else:
            print(f"   ❌ 未找到结果文件")
        
        print("\n" + "="*80 + "\n")
    
    def compare_sessions(self, session_id1, session_id2):
        """对比两个sessions"""
        sessions = self.list_sessions()
        
        print("\n" + "="*80)
        print(f"对比Sessions")
        print("="*80 + "\n")
        
        for i, sid in enumerate([session_id1, session_id2], 1):
            if sid not in sessions:
                print(f"❌ Session {sid} 不存在")
                continue
            
            info = sessions[sid]
            print(f"Session {i}: {sid}")
            
            if 'best_acc' in info:
                print(f"  最佳准确率: {info['best_acc']:.2f}%")
                print(f"  最佳Epoch: {info['best_epoch']}")
            
            if 'log' in info:
                print(f"  日志文件: {os.path.basename(info['log'])}")
            
            if 'checkpoints' in info:
                print(f"  模型数量: {len(info['checkpoints'])}")
            
            print()
        
        # 对比
        if session_id1 in sessions and session_id2 in sessions:
            info1 = sessions[session_id1]
            info2 = sessions[session_id2]
            
            if 'best_acc' in info1 and 'best_acc' in info2:
                diff = info1['best_acc'] - info2['best_acc']
                winner = session_id1 if diff > 0 else session_id2
                
                print(f"准确率差异: {abs(diff):.2f}%")
                print(f"更好的Session: {winner}")
        
        print("="*80 + "\n")
    
    def get_best_session(self):
        """获取最佳session"""
        sessions = self.list_sessions()
        
        best_session = None
        best_acc = 0.0
        
        for session_id, info in sessions.items():
            if 'best_acc' in info and info['best_acc'] > best_acc:
                best_acc = info['best_acc']
                best_session = session_id
        
        return best_session, best_acc
    
    def print_best_session(self):
        """打印最佳session"""
        best_session, best_acc = self.get_best_session()
        
        if best_session:
            print("\n" + "="*80)
            print(f"🏆 最佳Session: {best_session}")
            print(f"   准确率: {best_acc:.2f}%")
            print("="*80 + "\n")
            
            self.view_session(best_session)
        else:
            print("未找到任何session")
    
    def clean_old_sessions(self, keep_n=5, dry_run=True):
        """清理旧sessions，只保留最近N个"""
        sessions = self.list_sessions()
        
        # 按时间排序（session_id就是时间戳）
        sorted_sessions = sorted(sessions.keys(), reverse=True)
        
        if len(sorted_sessions) <= keep_n:
            print(f"只有 {len(sorted_sessions)} 个sessions，无需清理")
            return
        
        to_delete = sorted_sessions[keep_n:]
        
        print("\n" + "="*80)
        if dry_run:
            print(f"将要删除的Sessions（dry run）:")
        else:
            print(f"正在删除旧Sessions:")
        print("="*80 + "\n")
        
        for session_id in to_delete:
            info = sessions[session_id]
            print(f"Session: {session_id}")
            
            # 日志文件
            if 'log' in info and os.path.exists(info['log']):
                print(f"  删除: {info['log']}")
                if not dry_run:
                    os.remove(info['log'])
            
            # Checkpoint文件
            ckpt_files = glob.glob(f"{self.ckpt_dir}/{session_id}_*.pth")
            for f in ckpt_files:
                print(f"  删除: {f}")
                if not dry_run:
                    os.remove(f)
            
            print()
        
        if dry_run:
            print("注意: 这是dry run，使用 --no-dry-run 真正删除")
        else:
            print(f"✓ 已删除 {len(to_delete)} 个旧sessions")
        
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='训练Session管理工具')
    parser.add_argument('command', choices=['list', 'view', 'compare', 'clean', 'best'],
                       help='命令: list/view/compare/clean/best')
    parser.add_argument('args', nargs='*', help='命令参数')
    parser.add_argument('--log-dir', default='./log', help='日志目录')
    parser.add_argument('--ckpt-dir', default='./ckpt', help='checkpoint目录')
    parser.add_argument('--result-dir', default='./result', help='结果目录')
    parser.add_argument('--keep', type=int, default=5, help='clean命令保留的session数量')
    parser.add_argument('--no-dry-run', action='store_true', help='clean命令真正执行删除')
    
    args = parser.parse_args()
    
    manager = SessionManager(args.log_dir, args.ckpt_dir, args.result_dir)
    
    if args.command == 'list':
        manager.print_session_list()
    
    elif args.command == 'view':
        if not args.args:
            print("错误: 需要指定session_id")
            print("用法: python session_manager.py view <session_id>")
            return
        manager.view_session(args.args[0])
    
    elif args.command == 'compare':
        if len(args.args) < 2:
            print("错误: 需要指定两个session_id")
            print("用法: python session_manager.py compare <id1> <id2>")
            return
        manager.compare_sessions(args.args[0], args.args[1])
    
    elif args.command == 'clean':
        manager.clean_old_sessions(keep_n=args.keep, dry_run=not args.no_dry_run)
    
    elif args.command == 'best':
        manager.print_best_session()


if __name__ == '__main__':
    main()
