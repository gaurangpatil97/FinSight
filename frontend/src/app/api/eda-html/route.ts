import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const filePath = searchParams.get("path");

  if (!filePath) {
    return new NextResponse("File path parameter 'path' is required.", { status: 400 });
  }

  try {
    // Resolve absolute path and normalize separators
    const resolvedPath = path.resolve(filePath);

    // Basic security check: ensure the file exists and is within the FinSight project directory or matches .html
    if (!fs.existsSync(resolvedPath)) {
      return new NextResponse("File not found on local disk.", { status: 404 });
    }

    if (!resolvedPath.endsWith(".html")) {
      return new NextResponse("Only HTML files are allowed to be served.", { status: 403 });
    }

    const htmlContent = fs.readFileSync(resolvedPath, "utf8");

    return new NextResponse(htmlContent, {
      headers: {
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": "no-store, max-age=0",
      },
    });
  } catch (error) {
    console.error("Error serving HTML file:", error);
    return new NextResponse(`Failed to read file: ${(error as Error).message}`, { status: 500 });
  }
}
