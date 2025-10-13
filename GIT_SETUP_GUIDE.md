# Git 저장소 설정 가이드

DVS 프로젝트를 GitHub에 업로드하기 위한 단계별 가이드입니다.

## 🎯 현재 상황

- `filter_sim/.git` - 개별 git 저장소 존재
- `embeddedsw/.git` - 개별 git 저장소 존재 (제외 대상)
- 전체 `dvs/` 폴더를 하나의 저장소로 통합 필요

## 📋 제외할 항목 (.gitignore)

✅ **이미 설정 완료**:
- `embeddedsw/` - 전체 폴더
- `checkpoints*/` - 모든 체크포인트 폴더
- `*.bin` - DVS 데이터 파일 (용량이 큼)
- `__pycache__/`, `*.pyc` - Python 캐시
- `logs/` - 로그 폴더
- IDE 설정 파일 (`.vscode/`, `.idea/` 등)

## 🚀 방법 1: 자동 스크립트 사용 (권장)

```bash
cd /hai/home/jdj/dvs
./setup_git.sh
```

스크립트가 다음 작업을 자동으로 수행합니다:
1. ✅ 기존 서브 git 저장소 백업
2. ✅ 새 Git 저장소 초기화
3. ✅ 주요 파일 추가
4. ✅ 상태 확인

## 🔧 방법 2: 수동 설정

### Step 1: 기존 서브 저장소 백업

```bash
cd /hai/home/jdj/dvs

# filter_sim의 git 저장소 백업
mv filter_sim/.git filter_sim/.git.backup
```

### Step 2: Git 저장소 초기화

```bash
# dvs 폴더에서 새 저장소 초기화
git init
```

### Step 3: 파일 추가

```bash
# .gitignore 먼저 추가
git add .gitignore .gitattributes

# README 파일들 추가
git add README.md
git add filter_sim/README.md cnn_sim/README.md yolo_sim/README.md

# 의존성 파일 추가
git add requirements.txt environment.yml

# Python 파일 추가 (.gitignore가 자동으로 제외)
git add filter_sim/*.py
git add cnn_sim/*.py
git add yolo_sim/*.py
git add *.py

# 또는 모든 파일 추가 (.gitignore가 자동으로 필터링)
git add .
```

### Step 4: 상태 확인

```bash
git status
```

**확인할 사항**:
- ✅ `embeddedsw/` 폴더가 untracked에 없어야 함
- ✅ `checkpoints*/` 폴더가 없어야 함
- ✅ `*.bin` 파일이 없어야 함
- ✅ Python 소스 파일들이 staged 상태여야 함

### Step 5: 첫 커밋

```bash
git commit -m "Initial commit: DVS 레이저 중심점 감지 프로젝트

- Filter 기반 휴리스틱 방식 구현
- CNN 기반 딥러닝 방식 구현  
- YOLO 기반 객체 감지 방식 구현
- 3가지 방법론 비교 분석
- FPGA 구현을 위한 시뮬레이션 프레임워크
"
```

## 🌐 GitHub에 업로드

### Step 1: GitHub에서 새 저장소 생성

1. GitHub에 로그인
2. 새 저장소 생성 (예: `dvs-laser-tracking`)
3. **Initialize this repository with a README** 체크 해제 (이미 있음)

### Step 2: 원격 저장소 연결

```bash
# GitHub 저장소 URL을 자신의 URL로 변경
git remote add origin https://github.com/YOUR_USERNAME/dvs-laser-tracking.git

# 또는 SSH 사용
git remote add origin git@github.com:YOUR_USERNAME/dvs-laser-tracking.git
```

### Step 3: 브랜치 이름 확인 및 변경 (필요시)

```bash
# 현재 브랜치 확인
git branch

# main 브랜치로 변경 (GitHub 기본값)
git branch -M main
```

### Step 4: 푸시

```bash
# 첫 푸시
git push -u origin main
```

## ⚠️ 대용량 파일 처리

### 문제 상황

만약 `.bin` 파일이나 체크포인트를 이미 커밋했다면:

```bash
# 캐시 제거
git rm --cached -r checkpoints*
git rm --cached *.bin

# 커밋
git commit -m "Remove large files"
```

### Git LFS 사용 (선택적)

대용량 파일을 꼭 올려야 한다면 Git LFS 사용:

```bash
# Git LFS 설치 (Ubuntu/Debian)
sudo apt-get install git-lfs
git lfs install

# 대용량 파일 타입 추가
git lfs track "*.bin"
git lfs track "*.pth"

# .gitattributes 커밋
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

## 📊 제외된 파일 확인

현재 제외된 파일/폴더 목록 확인:

```bash
# .gitignore에 의해 무시되는 파일 확인
git status --ignored

# 또는
git clean -ndX
```

## 🔍 문제 해결

### 1. "filter_sim/.git" 서브모듈 경고

```bash
# filter_sim/.git 완전 제거
rm -rf filter_sim/.git
git add filter_sim/
```

### 2. 대용량 파일 푸시 실패

```bash
# 오류: "remote: error: File XXX is XX MB; this exceeds GitHub's file size limit"

# 해결: 히스토리에서 완전 제거
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch <파일경로>' \
  --prune-empty --tag-name-filter cat -- --all
```

### 3. 푸시 거부 (non-fast-forward)

```bash
# 원격 저장소 상태 확인
git fetch origin

# 로컬과 원격 병합
git pull origin main --rebase

# 다시 푸시
git push origin main
```

## 📁 최종 구조

```
dvs/                          # ✅ Git 저장소 루트
├── .git/                     # ✅ Git 폴더
├── .gitignore                # ✅ 제외 규칙
├── .gitattributes            # ✅ Git 설정
├── README.md                 # ✅ 메인 문서
├── requirements.txt          # ✅ 의존성
├── environment.yml           # ✅ Conda 환경
├── filter_sim/               # ✅ 포함
│   ├── .git.backup/          # ⚠️  백업 (선택적 삭제)
│   ├── README.md             # ✅ 포함
│   ├── *.py                  # ✅ 포함
│   └── checkpoints/          # ❌ 제외
├── cnn_sim/                  # ✅ 포함
│   ├── README.md             # ✅ 포함
│   ├── *.py                  # ✅ 포함
│   └── checkpoints*/         # ❌ 제외
├── yolo_sim/                 # ✅ 포함
│   ├── README.md             # ✅ 포함
│   ├── *.py                  # ✅ 포함
│   └── checkpoints*/         # ❌ 제외
├── data/
│   └── *.bin                 # ❌ 제외
└── embeddedsw/               # ❌ 전체 제외
```

## ✅ 체크리스트

- [ ] `.gitignore` 파일 생성 완료
- [ ] 기존 서브 저장소 백업
- [ ] Git 저장소 초기화 (`git init`)
- [ ] 파일 추가 및 상태 확인 (`git status`)
- [ ] `embeddedsw/`, `checkpoints*/`, `*.bin` 제외 확인
- [ ] 첫 커밋 완료
- [ ] GitHub에 새 저장소 생성
- [ ] 원격 저장소 연결 (`git remote add origin`)
- [ ] 푸시 완료 (`git push -u origin main`)
- [ ] GitHub에서 파일 확인

## 📞 추가 도움말

- Git 기초: https://git-scm.com/book/ko/v2
- GitHub 가이드: https://docs.github.com/ko
- Git LFS: https://git-lfs.github.com/

---

**작성일**: 2025-01-13  
**프로젝트**: DVS 레이저 중심점 감지 시스템

