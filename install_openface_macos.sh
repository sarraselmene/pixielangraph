#!/bin/bash
# =============================================================================
# install_openface_macos.sh (Version Corrigée & Robuste)
# =============================================================================
# Construit OpenFace 2.2.0 sur macOS (M1, M2, M3 ou Intel).
# Gère le bug de détection de Boost 1.90+ sur les systèmes récents.
# =============================================================================

set -e  # Arrête le script en cas d'erreur

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║      OpenFace macOS Build Script (Robust v2.0)       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 1. Détection de l'environnement ──────────────────────────────────────────
ARCH=$(uname -m)
BREW_PREFIX=$(brew --prefix)
echo "▶ Architecture détectée : $ARCH"
echo "▶ Homebrew prefix : $BREW_PREFIX"

# ── 2. Installation des dépendances ──────────────────────────────────────────
echo ""
echo "▶ Étape 1 : Installation des dépendances via Homebrew..."
brew install cmake boost tbb openblas opencv dlib wget git

# ── 3. Préparation du dossier OpenFace ───────────────────────────────────────
echo ""
echo "▶ Étape 2 : Clonage du dépôt OpenFace..."
INSTALL_DIR="$HOME/OpenFace"

if [ -d "$INSTALL_DIR" ]; then
    echo "  Le dossier $INSTALL_DIR existe déjà. Nettoyage de l'ancien build..."
    rm -rf "$INSTALL_DIR/build"
else
    git clone https://github.com/TadasBaltrusaitis/OpenFace.git "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# ── 4. Téléchargement des modèles ───────────────────────────────────────────
echo ""
echo "▶ Étape 3 : Téléchargement des modèles (Action Units & Landmarks)..."
if [ -f "download_models.sh" ]; then
    chmod +x download_models.sh
    ./download_models.sh
else
    echo "  ⚠️ Script download_models.sh manquant !"
    exit 1
fi

# ── 5. Configuration CMake (Correction du bug Boost) ─────────────────────────
echo ""
echo "▶ Étape 4 : Configuration avec CMake (Correction Boost 1.90)..."
mkdir -p build && cd build

# Astuce : On localise dynamiquement la bibliothèque filesystem pour corriger 
# le fait que libboost_system n'existe plus en tant que fichier dylib séparé.
BOOST_PATH=$(brew --prefix boost)
BOOST_LIB_DIR="$BOOST_PATH/lib"
# On cherche libboost_filesystem.dylib pour l'utiliser comme substitut
BOOST_FILESYSTEM_LIB=$(find "$BOOST_LIB_DIR" -name "libboost_filesystem.dylib" | head -n 1)

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBOOST_ROOT="$BOOST_PATH" \
    -DBoost_NO_BOOST_CMAKE=ON \
    -DBoost_SYSTEM_LIBRARY="$BOOST_FILESYSTEM_LIB" \
    -DBoost_FILESYSTEM_LIBRARY="$BOOST_FILESYSTEM_LIB" \
    -DOpenCV_DIR=$(brew --prefix opencv)/lib/cmake/opencv4 \
    -DOpenBLAS_INCLUDE_DIR="$BREW_PREFIX/opt/openblas/include" \
    -DOpenBLAS_LIB="$BREW_PREFIX/opt/openblas/lib/libopenblas.dylib" \
    -DCMAKE_PREFIX_PATH="$BREW_PREFIX" \
    -DCMAKE_CXX_FLAGS="-I$BREW_PREFIX/include"

# ── 6. Compilation ───────────────────────────────────────────────────────────
echo ""
echo "▶ Étape 5 : Compilation (Ceci peut prendre 10 minutes)..."
make -j$(sysctl -n hw.ncpu)

# ── 7. Organisation finale ────────────────────────────────────────────────────
echo ""
echo "▶ Étape 6 : Organisation des dossiers modèles..."
# On s'assure que les modèles sont au bon endroit pour les exécutables
mkdir -p bin/model
mkdir -p bin/AU_predictors
cp -r ../model/* bin/model/ 2>/dev/null || true
cp -r ../AU_predictors/* bin/AU_predictors/ 2>/dev/null || true

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║          ✅  INSTALLATION RÉUSSIE !                  ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Les logiciels sont dans :                           ║"
echo "║  ~/OpenFace/build/bin/                               ║"
echo "║                                                      ║"
echo "║  Testez avec :                                       ║"
echo "║  cd ~/OpenFace/build/bin && ./FaceLandmarkImg --help ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
