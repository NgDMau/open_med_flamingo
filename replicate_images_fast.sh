#!/bin/bash

# Fast image replication script using bash

SOURCE_IMAGE="/mnt/data/maund/open_med_flamingo/open_flamingo/data/images/sample_image.jpg"
TARGET_DIR="/mnt/data/maund/open_med_flamingo/open_flamingo/data/images"

echo "=== Fast Image Replication Script ==="
echo "Source: $SOURCE_IMAGE"
echo "Target: $TARGET_DIR"

# Check if source exists
if [ ! -f "$SOURCE_IMAGE" ]; then
    echo "❌ Source image not found: $SOURCE_IMAGE"
    exit 1
fi

# Check disk space
echo "📊 Checking disk space..."
FILE_SIZE=$(stat -c%s "$SOURCE_IMAGE")
TOTAL_SIZE=$((FILE_SIZE * 10000))
AVAILABLE=$(df "$TARGET_DIR" | awk 'NR==2 {print $4 * 1024}')

echo "  File size: $(($FILE_SIZE / 1024 / 1024)) MB"
echo "  Total needed: $(($TOTAL_SIZE / 1024 / 1024)) MB"
echo "  Available: $(($AVAILABLE / 1024 / 1024)) MB"

if [ $TOTAL_SIZE -gt $AVAILABLE ]; then
    echo "❌ Not enough disk space!"
    exit 1
fi

echo "✅ Sufficient disk space available"

# Ask for confirmation
echo ""
read -p "🤔 Create 10,000 copies? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Operation cancelled"
    exit 0
fi

echo "🔄 Creating 10,000 copies..."

# Create copies using a loop
cd "$TARGET_DIR"
for i in {1..10000}; do
    cp "$SOURCE_IMAGE" "${i}.jpg"
    
    # Progress update every 1000
    if [ $((i % 1000)) -eq 0 ]; then
        echo "  ✅ Copied $i images..."
    fi
done

echo ""
echo "🎉 Replication complete!"
echo "📊 Verification:"
echo "  Total files in directory: $(ls -1 *.jpg | wc -l)"
echo "  Sample files:"
ls -la 1.jpg 100.jpg 1000.jpg 5000.jpg 10000.jpg 2>/dev/null

echo ""
echo "✅ All done! You now have 10,001 images in the folder."
echo "💡 To clean up: rm {1..10000}.jpg"
