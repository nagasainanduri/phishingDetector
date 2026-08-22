import argparse
from detector import PhishingDetector

def main():
    parser = argparse.ArgumentParser(description='Phishing Web Detector CLI')
    parser.add_argument('--url', type=str, help='Single URL to analyze')
    parser.add_argument('--file', type=str, help='Path to a file with URLs (one per line)')
    args = parser.parse_args()

    try:
        detector = PhishingDetector()
    except Exception as e:
        print(f"Error initializing model: {e}")
        exit(1)

    if args.url:
        result = detector.analyze(args.url)
        print(f"URL: {result['url']}")
        print(f"Result: {result['result']}")
        print(f"Confidence: {result.get('confidence', 0)}%")
        if 'error' in result and result['error']:
            print(f"Error: {result['error']}")
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            for url in urls:
                result = detector.analyze(url)
                print(f"URL: {result['url']}")
                print(f"Result: {result['result']}")
                print(f"Confidence: {result.get('confidence', 0)}%")
                if 'error' in result and result['error']:
                    print(f"Error: {result['error']}")
                print("-" * 50)
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.")
    else:
        print("Error: Please provide a --url or --file argument.")
        parser.print_help()

if __name__ == '__main__':
    main()