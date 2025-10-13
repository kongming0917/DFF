#!/bin/bash
# DVS 프로젝트 Git 저장소 설정 스크립트

echo "🔧 DVS 프로젝트 Git 저장소 설정"
echo "================================"

cd /hai/home/jdj/dvs

# 1. 기존 서브 git 저장소 백업 (선택적)
echo ""
echo "📦 Step 1: 기존 git 저장소 백업"
if [ -d "filter_sim/.git" ]; then
    echo "   filter_sim/.git을 filter_sim/.git.backup으로 백업..."
    mv filter_sim/.git filter_sim/.git.backup
    echo "   ✅ 백업 완료"
else
    echo "   filter_sim/.git이 없습니다."
fi

if [ -d "embeddedsw/.git" ]; then
    echo "   embeddedsw/.git 발견 (이미 .gitignore에 포함됨)"
fi

# 2. Git 저장소 초기화
echo ""
echo "🆕 Step 2: Git 저장소 초기화"
if [ -d ".git" ]; then
    echo "   ⚠️  이미 .git 폴더가 존재합니다."
    read -p "   기존 저장소를 삭제하고 새로 초기화하시겠습니까? (y/N): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        rm -rf .git
        git init
        echo "   ✅ 새로운 Git 저장소 초기화 완료"
    else
        echo "   기존 저장소 유지"
    fi
else
    git init
    echo "   ✅ Git 저장소 초기화 완료"
fi

# 3. 파일 추가
echo ""
echo "➕ Step 3: 파일 추가"
git add .gitignore .gitattributes
git add README.md
git add filter_sim/README.md cnn_sim/README.md yolo_sim/README.md
git add requirements.txt environment.yml
git add "*.py"
echo "   ✅ 주요 파일 추가 완료"

# 4. 상태 확인
echo ""
echo "📊 Step 4: Git 상태 확인"
echo ""
git status

echo ""
echo "================================"
echo "✅ Git 저장소 설정 완료!"
echo ""
echo "💡 다음 단계:"
echo "   1. 추가할 파일 확인: git status"
echo "   2. 모든 파일 추가: git add ."
echo "   3. 첫 커밋: git commit -m 'Initial commit: DVS 레이저 중심점 감지 프로젝트'"
echo "   4. GitHub 원격 저장소 연결:"
echo "      git remote add origin <GitHub-저장소-URL>"
echo "   5. 푸시: git push -u origin main"
echo ""
echo "⚠️  주의사항:"
echo "   - .bin 파일은 자동으로 제외됩니다 (용량이 큼)"
echo "   - checkpoints 폴더도 제외됩니다 (모델 파일 용량)"
echo "   - embeddedsw 폴더는 제외됩니다"
echo "   - 대용량 파일은 Git LFS 사용을 권장합니다"
echo ""

